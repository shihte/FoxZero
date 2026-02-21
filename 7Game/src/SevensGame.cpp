#include "SevensGame.h"
#include <algorithm>
#include <random>
#include <stdexcept>

SevensGame::SevensGame() : SevensGame(4) {}

SevensGame::SevensGame(int numberOfPlayers)
    : numberOfPlayers(numberOfPlayers), currentPlayerNumber(1),
      firstMovePerformed(false) {
  setupNewGame();
}

void SevensGame::setupNewGame() {
  playedCards.clear();
  for (int i = 0; i < 4; ++i) {
    playedCards.push_back(std::make_unique<PlacedSuit>(i + 1));
  }
  performInitialDeal();
  firstMovePerformed = false;
  dealerNumber = findPlayerWithCard(Card::SPADE, 7);
  currentPlayerNumber = dealerNumber;

  passRecord.assign(numberOfPlayers, std::vector<bool>(4, false));
  coveredCards.assign(numberOfPlayers,
                      std::vector<Card>()); // Init covered cards
}

void SevensGame::performInitialDeal() {
  Deck deck(false);
  hands.assign(numberOfPlayers, Hand());

  int currentPlayerIndex = 0;
  while (deck.getNumberOfCardsRemaining() > 0) {
    hands[currentPlayerIndex].addCard(deck.dealCard());
    currentPlayerIndex = (currentPlayerIndex + 1) % numberOfPlayers;
  }
}

void SevensGame::nextPlayer() {
  currentPlayerNumber = (currentPlayerNumber % numberOfPlayers) + 1;
}

bool SevensGame::isValidMove(const Card &card) const {
  if (!firstMovePerformed) {
    return card.getSuit() == Card::SPADE && card.getRank() == 7;
  }
  int suitIndex = card.getSuit() - 1;
  if (suitIndex < 0 || suitIndex >= (int)playedCards.size())
    return false;
  return playedCards[suitIndex]->canCardBePlaced(card);
}

std::vector<Card> SevensGame::getAllValidMoves(int playerNumber) const {
  int playerIndex = playerNumber - 1;
  checkIsValidPlayerIndex(playerIndex);

  std::vector<Card> validMoves;
  for (const auto &card : hands[playerIndex].getHand()) {
    if (isValidMove(card)) {
      validMoves.push_back(card);
    }
  }

  // COVER RULE: If no valid moves on board, ALL cards in hand are valid for
  // covering.
  if (validMoves.empty()) {
    for (const auto &card : hands[playerIndex].getHand()) {
      validMoves.push_back(card);
    }
  }

  return validMoves;
}

bool SevensGame::hasPlayerWon(int playerNumber) const {
  int playerIndex = playerNumber - 1;
  checkIsValidPlayerIndex(playerIndex);
  return hands[playerIndex].getCardCount() == 0;
}

bool SevensGame::isFirstMove() const { return !firstMovePerformed; }

Hand &SevensGame::getPlayersHand(int playerNumber) {
  int playerIndex = playerNumber - 1;
  checkIsValidPlayerIndex(playerIndex);
  return hands[playerIndex];
}

const std::vector<Hand> &SevensGame::getHands() const { return hands; }

void SevensGame::makeMove(const Card &card) {
  if (isValidMove(card)) {
    removeCardFromPlayersHand(currentPlayerNumber, card);
    placeCard(card);
  } else {
    // If not valid on board, it's a cover move
    coverCard(card);
  }
  firstMovePerformed = true;
}

void SevensGame::coverCard(const Card &card) {
  int playerIndex = currentPlayerNumber - 1;
  checkIsValidPlayerIndex(playerIndex);

  // 1. Remove from hand
  hands[playerIndex].removeCard(card);

  // 2. Add to covered pile
  coveredCards[playerIndex].push_back(card);

  // 3. Record Pass (Implicit): Covering means you had no playable cards for any
  // suit. So we mark all currently playable suits as "passed" (missing) for
  // this player.
  recordPass(currentPlayerNumber);
}

void SevensGame::recordPass(int playerNumber) {
  int playerIndex = playerNumber - 1;
  checkIsValidPlayerIndex(playerIndex);
  // In Sevens, passing usually implies you can't play ANY of the current
  // playable suits. To match FoxZero design [Channels 6-8: 該花色已 Pass], we
  // record pass for all suits that currently have valid moves available in the
  // hand (but user passed?) Actually, FoxZero likely means "I don't have this
  // suit", so if I pass, any suit that had a playable card I MUST not have.
  // Let's simplify: Record pass for all suits where I *could* have played if I
  // had the card.
  for (int s = 1; s <= 4; ++s) {
    // If a suit has a playable slot but player passes, they lack THAT specific
    // card. We'll mark the suit as passed for this player.
    passRecord[playerIndex][s - 1] = true;
  }
}

SevensGame::GameResult SevensGame::calculateFinalRewards() const {
  GameResult result;
  result.rawPenalties.assign(numberOfPlayers, 0.0);
  result.normalizedRewards.assign(numberOfPlayers, 0.0);
  result.winnerNumber = 0;

  for (int i = 1; i <= numberOfPlayers; ++i) {
    if (hasPlayerWon(i)) {
      result.winnerNumber = i;
      break;
    }
  }

  double totalLossPool = 0;
  for (int i = 0; i < numberOfPlayers; ++i) {
    if (i + 1 == result.winnerNumber) {
      result.rawPenalties[i] = 0.0;
      continue;
    }

    // Points sum
    double score = 0;
    for (const auto &card : hands[i].getHand()) {
      score += card.getRank();
    }
    // Add covered cards to score
    for (const auto &card : coveredCards[i]) {
      score += card.getRank();
    }

    double multiplier = 1.0;
    if (result.winnerNumber != 0)
      multiplier *= 2.0; // Someone finished
    if (i + 1 == dealerNumber)
      multiplier *= 2.0; // Dealer lost
    if (score > 50)
      multiplier *= 2.0; // Threshold penalty

    result.rawPenalties[i] = score * multiplier;
    totalLossPool += result.rawPenalties[i];
  }

  const double SCALE_FACTOR = 800.0;
  for (int i = 0; i < numberOfPlayers; ++i) {
    if (i + 1 == result.winnerNumber) {
      result.normalizedRewards[i] = totalLossPool / SCALE_FACTOR;
    } else {
      result.normalizedRewards[i] = -result.rawPenalties[i] / SCALE_FACTOR;
    }
  }

  return result;
}

void SevensGame::placeCard(const Card &card) {
  int suitIndex = card.getSuit() - 1;
  int rank = card.getRank();
  PlacedSuit &ps = *playedCards[suitIndex];

  if (rank == 7) {
    ps.setLowestCard(card);
    ps.setHighestCard(card);
  } else if (rank > 7) {
    ps.setHighestCard(card);
  } else {
    ps.setLowestCard(card);
  }
}

void SevensGame::removeCardFromPlayersHand(int playerNumber, const Card &card) {
  int playerIndex = playerNumber - 1;
  checkIsValidPlayerIndex(playerIndex);
  hands[playerIndex].removeCard(card);
}

void SevensGame::checkIsValidPlayerIndex(int playerIndex) const {
  if (playerIndex < 0 || playerIndex >= numberOfPlayers) {
    throw std::invalid_argument("Invalid Player Number");
  }
}

int SevensGame::findPlayerWithCard(int suit, int rank) const {
  for (int i = 0; i < numberOfPlayers; ++i) {
    if (hands[i].getCard(suit, rank).has_value()) {
      return i + 1;
    }
  }
  return 1;
}

int SevensGame::getNumberOfPlayers() const { return numberOfPlayers; }
int SevensGame::getCurrentPlayerNumber() const { return currentPlayerNumber; }
int SevensGame::getDealerNumber() const { return dealerNumber; }

const std::vector<std::unique_ptr<PlacedSuit>> &
SevensGame::getPlayedCards() const {
  return playedCards;
}

const std::vector<std::vector<bool>> &SevensGame::getPassRecord() const {
  return passRecord;
}

const std::vector<std::vector<Card>> &SevensGame::getCoveredCards() const {
  return coveredCards;
}

// Custom copy constructor for deep cloning
SevensGame::SevensGame(const SevensGame &other)
    : numberOfPlayers(other.numberOfPlayers),
      currentPlayerNumber(other.currentPlayerNumber),
      dealerNumber(other.dealerNumber),
      firstMovePerformed(other.firstMovePerformed), turnCount(other.turnCount),
      hands(other.hands), coveredCards(other.coveredCards),
      passRecord(other.passRecord) {
  for (const auto &ps : other.playedCards) {
    auto new_ps = std::make_unique<PlacedSuit>(ps->getSuit());
    if (ps->getLowestCard().has_value())
      new_ps->setLowestCard(ps->getLowestCard().value());
    if (ps->getHighestCard().has_value())
      new_ps->setHighestCard(ps->getHighestCard().value());
    playedCards.push_back(std::move(new_ps));
  }
}

SevensGame &SevensGame::operator=(const SevensGame &other) {
  if (this == &other)
    return *this;
  numberOfPlayers = other.numberOfPlayers;
  currentPlayerNumber = other.currentPlayerNumber;
  dealerNumber = other.dealerNumber;
  firstMovePerformed = other.firstMovePerformed;
  turnCount = other.turnCount;
  hands = other.hands;
  coveredCards = other.coveredCards;
  passRecord = other.passRecord;
  playedCards.clear();
  for (const auto &ps : other.playedCards) {
    auto new_ps = std::make_unique<PlacedSuit>(ps->getSuit());
    if (ps->getLowestCard().has_value())
      new_ps->setLowestCard(ps->getLowestCard().value());
    if (ps->getHighestCard().has_value())
      new_ps->setHighestCard(ps->getHighestCard().value());
    playedCards.push_back(std::move(new_ps));
  }
  return *this;
}

std::unique_ptr<SevensGame> SevensGame::clone() const {
  return std::make_unique<SevensGame>(*this);
}

void SevensGame::setFirstMovePerformed(bool val) { firstMovePerformed = val; }
void SevensGame::setCurrentPlayer(int playerNum) {
  currentPlayerNumber = playerNum;
}
void SevensGame::setDealer(int playerNum) { dealerNumber = playerNum; }

void SevensGame::setHand(int playerNum, const std::vector<Card> &cards) {
  checkIsValidPlayerIndex(playerNum - 1);
  hands[playerNum - 1] = Hand();
  for (const auto &c : cards) {
    hands[playerNum - 1].addCard(c);
  }
}

void SevensGame::setCoveredCards(int playerNum,
                                 const std::vector<Card> &cards) {
  checkIsValidPlayerIndex(playerNum - 1);
  coveredCards[playerNum - 1] = cards;
}

void SevensGame::setPlayedCardRange(int suit, int lowRank, int highRank) {
  if (suit < 1 || suit > 4)
    return;
  auto new_ps = std::make_unique<PlacedSuit>(suit);
  if (lowRank > 0 && highRank > 0) {
    new_ps->setLowestCard(Card(suit, lowRank));
    new_ps->setHighestCard(Card(suit, highRank));
  }
  playedCards[suit - 1] = std::move(new_ps);
}

int SevensGame::getTurnCount() const { return turnCount; }
void SevensGame::setTurnCount(int count) { turnCount = count; }

void SevensGame::determinize(int observerPlayer,
                             const std::vector<float> &beliefProbs) {
  int observerIdx = observerPlayer - 1;
  std::vector<int> opponentIndices;
  for (int i = 0; i < numberOfPlayers; ++i) {
    if (i != observerIdx)
      opponentIndices.push_back(i);
  }

  std::vector<Card> unknownCards;
  std::vector<int> handCounts(numberOfPlayers, 0);
  std::vector<int> coveredCounts(numberOfPlayers, 0);

  for (int i : opponentIndices) {
    handCounts[i] = hands[i].getCardCount();
    coveredCounts[i] = coveredCards[i].size();

    for (const auto &c : hands[i].getHand())
      unknownCards.push_back(c);
    for (const auto &c : coveredCards[i])
      unknownCards.push_back(c);

    hands[i] = Hand();
    coveredCards[i].clear();
  }

  if (unknownCards.empty())
    return;

  std::random_device rd;
  std::mt19937 g(rd());

  int max_attempts = 10;
  bool success = false;

  for (int attempt = 0; attempt < max_attempts; ++attempt) {
    std::vector<Card> pool = unknownCards;
    bool possible = true;

    std::vector<Hand> tempHands(numberOfPlayers);
    std::vector<std::vector<Card>> tempCovered(numberOfPlayers);

    // If beliefProbs is empty, shuffle globally at start of attempt
    if (beliefProbs.empty()) {
      std::shuffle(pool.begin(), pool.end(), g);
    }

    for (int i : opponentIndices) {
      if (!possible)
        break;

      int num_cov = coveredCounts[i];
      for (int k = 0; k < num_cov; ++k) {
        if (pool.empty()) {
          possible = false;
          break;
        }
        // For covered cards, we just pick uniformly since belief targets
        // strictly "hand" presence.
        std::uniform_int_distribution<> dis(0, pool.size() - 1);
        int pick = dis(g);
        tempCovered[i].push_back(pool[pick]);
        pool.erase(pool.begin() + pick);
      }

      int num_hand = handCounts[i];
      for (int k = 0; k < num_hand; ++k) {
        if (pool.empty()) {
          possible = false;
          break;
        }

        if (!beliefProbs.empty() && beliefProbs.size() == 3 * 4 * 13) {
          // Weighted sample based on beliefProbs
          // The tensor relates to specific player offset
          int offset = 1;
          if (i == (observerIdx + 2) % 4)
            offset = 2;
          if (i == (observerIdx + 3) % 4)
            offset = 3;

          std::vector<float> weights;
          for (const auto &c : pool) {
            int s_idx = c.getSuit() - 1;
            int r_idx = c.getRank() - 1;
            // Flat index inside 3x4x13
            int flat_idx = (offset - 1) * 52 + s_idx * 13 + r_idx;
            // Epsilon to ensure no strictly 0 Probability blocks distribution
            weights.push_back(beliefProbs[flat_idx] + 0.001f);
          }
          std::discrete_distribution<int> ddist(weights.begin(), weights.end());
          int pick = ddist(g);
          tempHands[i].addCard(pool[pick]);
          pool.erase(pool.begin() + pick);
        } else {
          // Without valid beliefs or weights just take from end (shuffled)
          tempHands[i].addCard(pool.back());
          pool.pop_back();
        }
      }
    }

    if (possible) {
      for (int i : opponentIndices) {
        hands[i] = tempHands[i];
        coveredCards[i] = tempCovered[i];
      }
      success = true;
      break;
    }
  }

  if (!success) {
    // Fallback: Uniform random distribution
    std::shuffle(unknownCards.begin(), unknownCards.end(), g);
    int current_idx = 0;
    for (int i : opponentIndices) {
      int num_cov = coveredCounts[i];
      for (int k = 0; k < num_cov; ++k)
        coveredCards[i].push_back(unknownCards[current_idx++]);

      int num_hand = handCounts[i];
      for (int k = 0; k < num_hand; ++k)
        hands[i].addCard(unknownCards[current_idx++]);
    }
  }
}
