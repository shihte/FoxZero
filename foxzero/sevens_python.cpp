#include "../7Game/include/Card.h"
#include "../7Game/include/Hand.h"
#include "../7Game/include/PlacedSuit.h"
#include "../7Game/include/SevensGame.h"
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <pybind11/numpy.h>
#include <thread>

namespace py = pybind11;

struct InferenceMCTSNode {
  InferenceMCTSNode *parent = nullptr;
  std::map<int, InferenceMCTSNode *> children;
  std::atomic<int> visits{0};
  std::atomic<float> value_sum{0.0f};
  std::atomic<int> virtual_loss{0};
  float prior = 0.0f;
  std::mutex node_mtx;

  InferenceMCTSNode(InferenceMCTSNode *p = nullptr, float pr = 0.0f)
      : parent(p), prior(pr) {}

  ~InferenceMCTSNode() {
    for (auto &pair : children) {
      delete pair.second;
    }
  }

  void add_value(float v) {
    float expected = value_sum.load(std::memory_order_relaxed);
    while (!value_sum.compare_exchange_weak(expected, expected + v,
                                            std::memory_order_release,
                                            std::memory_order_relaxed)) {
    }
  }

  float ucb_score(float c_puct) {
    int v_loss = virtual_loss.load(std::memory_order_relaxed);
    int v = visits.load(std::memory_order_relaxed) + v_loss;
    float val = value_sum.load(std::memory_order_relaxed) - v_loss;

    int p_v = 1;
    if (parent) {
      p_v = parent->visits.load(std::memory_order_relaxed) +
            parent->virtual_loss.load(std::memory_order_relaxed);
      if (p_v == 0)
        p_v = 1;
    }

    if (!parent || v == 0)
      return std::numeric_limits<float>::infinity();

    float q = val / v;
    float u = c_puct * prior * std::sqrt((float)p_v) / (1.0f + v);
    return q + u;
  }
};

std::vector<float> get_observation_internal(const SevensGame &g,
                                            int observer_player) {
  std::vector<float> tensor(11 * 4 * 13, 0.0f);

  auto set_val = [&](int ch, int row, int col, float val) {
    if (ch >= 0 && ch < 11 && row >= 0 && row < 4 && col >= 0 && col < 13) {
      tensor[ch * 52 + row * 13 + col] = val;
    }
  };

  int player_idx = observer_player - 1;
  int num_players = g.getNumberOfPlayers();

  const auto &my_hand = g.getHands()[player_idx];
  for (const auto &card : my_hand.getHand()) {
    set_val(0, card.getSuit() - 1, card.getRank() - 1, 1.0f);
  }

  const auto &played = g.getPlayedCards();
  for (int s = 0; s < 4; ++s) {
    auto lowest = played[s]->getLowestCard();
    auto highest = played[s]->getHighestCard();
    if (lowest.has_value() && highest.has_value()) {
      int min_r = lowest->getRank();
      int max_r = highest->getRank();
      for (int r = min_r; r <= max_r; ++r) {
        set_val(1, s, r - 1, 1.0f);
      }
    }
  }

  auto valid_moves = g.getAllValidMoves(observer_player);
  for (const auto &c : valid_moves) {
    set_val(2, c.getSuit() - 1, c.getRank() - 1, 1.0f);
  }

  for (int offset = 1; offset < 4; ++offset) {
    int opp_idx = (player_idx + offset) % num_players;
    int count = g.getHands()[opp_idx].getCardCount();
    float val = (float)count / 13.0f;
    for (int s = 0; s < 4; ++s) {
      for (int r = 0; r < 13; ++r) {
        set_val(2 + offset, s, r, val);
      }
    }
  }

  // Ch 6-8: Opponent Pass Record (Suit-level missing information)
  const auto &pass_record = g.getPassRecord();
  for (int offset = 1; offset < 4; ++offset) {
    int opp_idx = (player_idx + offset) % num_players;
    const auto &opp_passes = pass_record[opp_idx];
    for (int s = 0; s < 4; ++s) {
      if (opp_passes[s]) {
        for (int r = 0; r < 13; ++r) {
          set_val(5 + offset, s, r, 1.0f);
        }
      }
    }
  }

  if (observer_player == g.getDealerNumber()) {
    for (int i = 0; i < 52; ++i)
      tensor[9 * 52 + i] = 1.0f;
  }

  int total_cards_held = 0;
  for (const auto &h : g.getHands())
    total_cards_held += h.getCardCount();
  int turn_count = 52 - total_cards_held;
  float turn_val = (float)turn_count / 52.0f;
  for (int i = 0; i < 52; ++i)
    tensor[10 * 52 + i] = turn_val;

  return tensor;
}

struct MCTSBatchState {
  int num_threads;
  std::vector<float> batch_obs;
  std::vector<float> batch_policy;
  std::vector<float> batch_value;
  std::vector<bool> needs_inference;

  int waiting_for_inference = 0;
  int active_workers = 0;

  std::mutex mtx;
  std::condition_variable cv_main;
  std::condition_variable cv_workers;
};

int run_mcts_cpp_multithreaded(const SevensGame &original_game,
                               py::object py_model_predict_batch_fn,
                               int simulations, float c_puct, bool god_mode,
                               int num_threads) {
  int current_player = original_game.getCurrentPlayerNumber();
  auto valid_moves = original_game.getAllValidMoves(current_player);
  if (valid_moves.empty())
    return -1;
  if (valid_moves.size() == 1) {
    return (valid_moves[0].getSuit() - 1) * 13 + (valid_moves[0].getRank() - 1);
  }

  InferenceMCTSNode root;
  std::vector<float> root_belief(3 * 4 * 13, 0.0f);

  {
    std::vector<float> root_obs =
        get_observation_internal(original_game, current_player);
    py::array_t<float> root_obs_arr({1, 11, 4, 13});
    std::memcpy(root_obs_arr.mutable_data(), root_obs.data(),
                root_obs.size() * sizeof(float));

    py::object pr_tuple = py_model_predict_batch_fn(root_obs_arr);
    py::tuple res_tup = pr_tuple.cast<py::tuple>();
    py::array_t<float> p_dist = res_tup[0].cast<py::array_t<float>>();
    auto p_ptr = (const float *)p_dist.request().ptr;

    py::array_t<float> b_dist = res_tup[2].cast<py::array_t<float>>();
    auto b_ptr = (const float *)b_dist.request().ptr;
    root_belief.assign(b_ptr, b_ptr + (3 * 4 * 13));

    for (const auto &card : valid_moves) {
      int idx = (card.getSuit() - 1) * 13 + (card.getRank() - 1);
      root.children[idx] = new InferenceMCTSNode(&root, p_ptr[idx]);
    }
  }

  MCTSBatchState bstate;
  bstate.num_threads = num_threads;
  bstate.batch_obs.resize(num_threads * 11 * 4 * 13, 0.0f);
  bstate.batch_policy.resize(num_threads * 52, 0.0f);
  bstate.batch_value.resize(num_threads, 0.0f);
  bstate.needs_inference.resize(num_threads, false);
  bstate.active_workers = num_threads;

  int sims_per_thread = simulations / num_threads;

  auto worker_func = [&](int thread_id) {
    for (int i = 0; i < sims_per_thread; ++i) {
      InferenceMCTSNode *node = &root;
      auto scratch_game = original_game.clone();
      if (!god_mode) {
        scratch_game->determinize(current_player, root_belief);
      }

      std::vector<InferenceMCTSNode *> path;
      path.push_back(node);

      while (true) {
        int current_p = scratch_game->getCurrentPlayerNumber();
        auto v_moves = scratch_game->getAllValidMoves(current_p);
        if (v_moves.empty())
          break;

        std::vector<int> feasible;
        std::vector<int> unexpanded;

        node->node_mtx.lock();
        for (const auto &c : v_moves) {
          int c_idx = (c.getSuit() - 1) * 13 + (c.getRank() - 1);
          if (node->children.find(c_idx) != node->children.end()) {
            feasible.push_back(c_idx);
          } else {
            unexpanded.push_back(c_idx);
          }
        }
        node->node_mtx.unlock();

        if (!unexpanded.empty())
          break;

        if (!feasible.empty()) {
          int best_card = -1;
          float best_ucb = -std::numeric_limits<float>::infinity();
          for (int c_idx : feasible) {
            float s = node->children[c_idx]->ucb_score(c_puct);
            if (s > best_ucb) {
              best_ucb = s;
              best_card = c_idx;
            }
          }

          if (best_card != -1) {
            InferenceMCTSNode *next_node = node->children[best_card];
            next_node->virtual_loss.fetch_add(1, std::memory_order_relaxed);
            node = next_node;
            path.push_back(node);
            scratch_game->makeMove(
                Card(best_card / 13 + 1, best_card % 13 + 1));
            scratch_game->nextPlayer();
          } else {
            break;
          }
        } else {
          break;
        }
      }

      float value = 0.0f;
      int leaf_player = scratch_game->getCurrentPlayerNumber();
      bool is_game_over = true;
      for (const auto &h : scratch_game->getHands()) {
        if (h.getCardCount() > 0) {
          is_game_over = false;
          break;
        }
      }

      if (is_game_over) {
        auto rewards = scratch_game->calculateFinalRewards();
        float r = rewards.normalizedRewards[leaf_player - 1];
        if (r > 0)
          value = 1.0f;
        else if (r < 0)
          value = -1.0f;
        else
          value = 0.0f;
      } else {
        leaf_player = scratch_game->getCurrentPlayerNumber();
        auto v_moves = scratch_game->getAllValidMoves(leaf_player);
        if (!v_moves.empty()) {
          std::vector<float> obs =
              get_observation_internal(*scratch_game, leaf_player);

          std::unique_lock<std::mutex> lk(bstate.mtx);
          std::memcpy(&bstate.batch_obs[thread_id * 11 * 4 * 13], obs.data(),
                      11 * 4 * 13 * sizeof(float));
          bstate.needs_inference[thread_id] = true;
          bstate.waiting_for_inference++;

          if (bstate.waiting_for_inference == bstate.active_workers) {
            bstate.cv_main.notify_one();
          }

          bstate.cv_workers.wait(
              lk, [&]() { return !bstate.needs_inference[thread_id]; });

          value = bstate.batch_value[thread_id];
          std::vector<float> p_dist(&bstate.batch_policy[thread_id * 52],
                                    &bstate.batch_policy[(thread_id + 1) * 52]);
          lk.unlock();

          node->node_mtx.lock();
          for (const auto &c : v_moves) {
            int c_idx = (c.getSuit() - 1) * 13 + (c.getRank() - 1);
            if (node->children.find(c_idx) == node->children.end()) {
              node->children[c_idx] =
                  new InferenceMCTSNode(node, p_dist[c_idx]);
            }
          }
          node->node_mtx.unlock();
        }
      }

      float curr_val = value;
      for (int k = path.size() - 1; k >= 0; --k) {
        InferenceMCTSNode *curr = path[k];
        if (k > 0) {
          curr->virtual_loss.fetch_sub(1, std::memory_order_relaxed);
        }
        curr->add_value(curr_val);
        curr->visits.fetch_add(1, std::memory_order_relaxed);
        curr_val = -curr_val;
      }

      if (thread_id == 0 && (i + 1) % 10 == 0) {
        std::cout << "\r🔍 MCTS 多執行緒運算中... " << (i + 1) * num_threads
                  << "/" << simulations << std::flush;
      }
    }

    std::unique_lock<std::mutex> lk(bstate.mtx);
    bstate.active_workers--;
    if (bstate.waiting_for_inference == bstate.active_workers) {
      bstate.cv_main.notify_one();
    }
  };

  std::vector<std::thread> threads;

  {
    py::gil_scoped_release release;
    for (int i = 0; i < num_threads; ++i) {
      threads.emplace_back(worker_func, i);
    }

    while (true) {
      std::unique_lock<std::mutex> lk(bstate.mtx);
      bstate.cv_main.wait(lk, [&]() {
        return bstate.waiting_for_inference == bstate.active_workers ||
               bstate.active_workers == 0;
      });

      if (bstate.active_workers == 0 && bstate.waiting_for_inference == 0) {
        break;
      }

      if (bstate.waiting_for_inference > 0) {
        int current_batch_size = 0;
        std::vector<int> active_thread_ids;
        for (int i = 0; i < num_threads; ++i) {
          if (bstate.needs_inference[i]) {
            active_thread_ids.push_back(i);
            current_batch_size++;
          }
        }

        std::vector<float> dense_obs(current_batch_size * 11 * 4 * 13);
        for (int v = 0; v < current_batch_size; ++v) {
          int tid = active_thread_ids[v];
          std::memcpy(&dense_obs[v * 11 * 4 * 13],
                      &bstate.batch_obs[tid * 11 * 4 * 13],
                      11 * 4 * 13 * sizeof(float));
        }

        lk.unlock();

        py::gil_scoped_acquire acquire;
        py::array_t<float> batch_arr({current_batch_size, 11, 4, 13});
        std::memcpy(batch_arr.mutable_data(), dense_obs.data(),
                    dense_obs.size() * sizeof(float));

        py::object pr_tuple = py_model_predict_batch_fn(batch_arr);
        py::tuple res_tup = pr_tuple.cast<py::tuple>();

        py::array_t<float> p_dists = res_tup[0].cast<py::array_t<float>>();
        py::array_t<float> vals = res_tup[1].cast<py::array_t<float>>();
        const float *p_ptr_batch = (const float *)p_dists.request().ptr;
        const float *v_ptr_batch = (const float *)vals.request().ptr;

        py::gil_scoped_release release_again;

        lk.lock();
        for (int v = 0; v < current_batch_size; ++v) {
          int tid = active_thread_ids[v];
          std::memcpy(&bstate.batch_policy[tid * 52], &p_ptr_batch[v * 52],
                      52 * sizeof(float));
          bstate.batch_value[tid] = v_ptr_batch[v];
          bstate.needs_inference[tid] = false;
        }
        bstate.waiting_for_inference = 0;
        bstate.cv_workers.notify_all();
      }
    }
  }

  for (auto &t : threads)
    t.join();

  std::cout << "\n";

  int best_card = -1;
  int max_visits = -1;
  for (const auto &pair : root.children) {
    if (pair.second->visits.load() > max_visits) {
      max_visits = pair.second->visits.load();
      best_card = pair.first;
    }
  }

  return best_card;
}

PYBIND11_MODULE(sevens_core, m) {
  m.doc() = "Sevens Card Game C++ Core";

  py::class_<Card>(m, "Card")
      .def(py::init<int, int>())
      .def("getSuit", &Card::getSuit)
      .def("getRank", &Card::getRank)
      .def("__repr__", &Card::toString)
      .def("to_tensor_index",
           [](const Card &c) {
             return std::make_pair(c.getSuit() - 1, c.getRank() - 1);
           })
      .def_property_readonly("suit", &Card::getSuit)
      .def_property_readonly("rank", &Card::getRank);

  py::class_<Hand>(m, "Hand")
      .def("getHand", &Hand::getHand)
      .def("getCardCount", &Hand::getCardCount);

  py::class_<SevensGame::GameResult>(m, "GameResult")
      .def_readonly("rawPenalties", &SevensGame::GameResult::rawPenalties)
      .def_readonly("normalizedRewards",
                    &SevensGame::GameResult::normalizedRewards)
      .def_readonly("winnerNumber", &SevensGame::GameResult::winnerNumber);

  py::class_<SevensGame>(m, "SevensEngine")
      .def(py::init<>())
      .def(py::init<int>())
      .def("setupNewGame", &SevensGame::setupNewGame)
      .def("setFirstMovePerformed", &SevensGame::setFirstMovePerformed)
      .def("setCurrentPlayer", &SevensGame::setCurrentPlayer)
      .def("setDealer", &SevensGame::setDealer)
      .def("setTurnCount", &SevensGame::setTurnCount)
      .def("setHand", &SevensGame::setHand)
      .def("setPlayedCardRange", &SevensGame::setPlayedCardRange)
      .def("setCoveredCards", &SevensGame::setCoveredCards)
      .def("nextPlayer", &SevensGame::nextPlayer)
      .def("isValidMove", &SevensGame::isValidMove)
      .def("getAllValidMoves", &SevensGame::getAllValidMoves)
      .def("makeMove", &SevensGame::makeMove)
      .def("coverCard", &SevensGame::coverCard)
      .def("hasPlayerWon", &SevensGame::hasPlayerWon)
      .def("isFirstMove", &SevensGame::isFirstMove)
      .def("getPlayersHand", &SevensGame::getPlayersHand,
           py::return_value_policy::reference)
      .def("getHands", &SevensGame::getHands,
           py::return_value_policy::reference)
      .def("getCurrentPlayerNumber", &SevensGame::getCurrentPlayerNumber)
      .def("calculateFinalRewards", &SevensGame::calculateFinalRewards)
      .def("getPassRecord", &SevensGame::getPassRecord)
      .def("isGameOver",
           [](const SevensGame &g) {
             for (const auto &h : g.getHands()) {
               if (h.getCardCount() > 0)
                 return false;
             }
             return true;
           })
      .def("get_legal_moves",
           [](const SevensGame &g) {
             std::vector<float> mask(52, 0.0f);
             auto valid = g.getAllValidMoves(g.getCurrentPlayerNumber());
             for (const auto &c : valid) {
               int idx = (c.getSuit() - 1) * 13 + (c.getRank() - 1);
               if (idx >= 0 && idx < 52)
                 mask[idx] = 1.0f;
             }
             return mask;
           })
      .def("get_observation",
           [](const SevensGame &g, int observer_player) {
             std::vector<float> obs =
                 get_observation_internal(g, observer_player);
             py::array_t<float> result({11 * 4 * 13});
             std::memcpy(result.mutable_data(), obs.data(),
                         obs.size() * sizeof(float));
             return result;
           })
      .def("get_belief_target",
           [](const SevensGame &g, int observer_player) {
             std::vector<float> target(3 * 4 * 13, 0.0f);
             int obs_idx = observer_player - 1;

             std::vector<int> opp_indices;
             for (int i = 0; i < 4; ++i) {
               if (i != obs_idx)
                 opp_indices.push_back(i);
             }

             for (int opp = 0; opp < 3; ++opp) {
               int p_idx = opp_indices[opp];

               // Hand
               const auto &hand = g.getHands()[p_idx].getHand();
               for (const auto &c : hand) {
                 int s = c.getSuit() - 1;
                 int r = c.getRank() - 1;
                 target[opp * 52 + s * 13 + r] = 1.0f;
               }

               // Covered Cards
               const auto &covered = g.getCoveredCards()[p_idx];
               for (const auto &c : covered) {
                 int s = c.getSuit() - 1;
                 int r = c.getRank() - 1;
                 target[opp * 52 + s * 13 + r] = 1.0f;
               }
             }

             py::array_t<float> result({3 * 4 * 13});
             std::memcpy(result.mutable_data(), target.data(),
                         target.size() * sizeof(float));
             return result;
           })
      .def("step", [](SevensGame &g, int action_idx) {
        int s = action_idx / 13;
        int r = action_idx % 13;
        Card c(s + 1, r + 1);
        g.makeMove(c);
        g.nextPlayer();
      });

  m.def("run_mcts_cpp", &run_mcts_cpp_multithreaded, py::arg("original_game"),
        py::arg("py_model_predict_batch_fn"), py::arg("simulations"),
        py::arg("c_puct") = 1.0f, py::arg("god_mode") = false,
        py::arg("num_threads") = 8);
}
