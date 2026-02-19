#include "../7Game/include/Card.h"
#include "../7Game/include/Hand.h"
#include "../7Game/include/PlacedSuit.h"
#include "../7Game/include/SevensGame.h"
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

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
             // 11x4x13 Tensor flattened
             std::vector<float> tensor(11 * 4 * 13, 0.0f);

             auto set_val = [&](int ch, int row, int col, float val) {
               if (ch >= 0 && ch < 11 && row >= 0 && row < 4 && col >= 0 &&
                   col < 13) {
                 tensor[ch * 52 + row * 13 + col] = val;
               }
             };

             int player_idx = observer_player - 1; // 0-based
             int num_players = g.getNumberOfPlayers();

             // Ch 0: My Hand
             const auto &my_hand = g.getHands()[player_idx];
             for (const auto &card : my_hand.getHand()) {
               set_val(0, card.getSuit() - 1, card.getRank() - 1, 1.0f);
             }

             // Ch 1: Board State
             const auto &played = g.getPlayedCards();
             for (int s = 0; s < 4; ++s) { // Suits 0-3
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

             // Ch 2: Legal Moves (My valid moves)
             // Note: In FoxZero common.py logic, this isn't explicitly used as
             // input feature usually? Wait, common.py logic for State
             // Representation: Ch 2: Legal Moves. Yes, game.py has it.
             auto valid_moves = g.getAllValidMoves(observer_player);
             for (const auto &c : valid_moves) {
               set_val(2, c.getSuit() - 1, c.getRank() - 1, 1.0f);
             }

             // Ch 3-5: Opponent Hand Count (Normalized) -> Plane
             for (int offset = 1; offset < 4; ++offset) {
               int opp_idx = (player_idx + offset) % num_players;
               int count = g.getHands()[opp_idx].getCardCount();
               float val = (float)count / 13.0f;
               // Fill entire plane for this channel
               for (int s = 0; s < 4; ++s) {
                 for (int r = 0; r < 13; ++r) {
                   set_val(2 + offset, s, r, val);
                 }
               }
             }

             // Ch 6-8: Opponent Pass Record -> Row per suit
             const auto &pass_rec = g.getPassRecord();
             for (int offset = 1; offset < 4; ++offset) {
               int opp_idx = (player_idx + offset) % num_players;
               const auto &rec = pass_rec[opp_idx]; // vector<bool> size 4
               for (int s = 0; s < 4; ++s) {
                 if (rec[s]) {
                   // Fill row for this suit
                   for (int r = 0; r < 13; ++r) {
                     set_val(5 + offset, s, r, 1.0f);
                   }
                 }
               }
             }

             // Ch 9: Dealer Indicator
             if (observer_player == g.getDealerNumber()) {
               for (int i = 0; i < 52; ++i)
                 tensor[9 * 52 + i] = 1.0f;
             }

             // Ch 10: Turn Count
             int total_cards_held = 0;
             for (const auto &h : g.getHands())
               total_cards_held += h.getCardCount();
             int turn_count = 52 - total_cards_held; // Approx
             float turn_val = (float)turn_count / 52.0f;
             for (int i = 0; i < 52; ++i)
               tensor[10 * 52 + i] = turn_val;

             return tensor;
           })
      .def("step", [](SevensGame &g, int action_idx) {
        // Action index 0-51 -> Card
        int s = action_idx / 13;
        int r = action_idx % 13;
        Card c(s + 1, r + 1);
        g.makeMove(c);
        g.nextPlayer();
      });
}
