#include "SevensGame.h"
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
