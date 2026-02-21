#ifndef SEVENSGAME_H
#define SEVENSGAME_H

#include "Card.h"
#include "Deck.h"
#include "Hand.h"
#include "PlacedSuit.h"
#include <memory>
#include <stdexcept>
#include <vector>

class SevensGame {
public:
  SevensGame();
  SevensGame(int numberOfPlayers);

  // Custom copy constructor for deep cloning
  SevensGame(const SevensGame &other);
  SevensGame &operator=(const SevensGame &other);
  std::unique_ptr<SevensGame> clone() const;

  void setupNewGame();
  void nextPlayer();

  bool isValidMove(const Card &card) const;
  std::vector<Card> getAllValidMoves(int playerNumber) const;

  bool hasPlayerWon(int playerNumber) const;
  bool isFirstMove() const;
  Hand &getPlayersHand(int playerNumber);
  const std::vector<Hand> &getHands() const;

  void makeMove(const Card &card);
  void recordPass(int playerNumber);

  struct GameResult {
    std::vector<double> rawPenalties;
    std::vector<double> normalizedRewards;
    int winnerNumber; // 1-based, 0 if draw
  };
  GameResult calculateFinalRewards() const;

  int getNumberOfPlayers() const;
  int getCurrentPlayerNumber() const;
  int getDealerNumber() const;
  const std::vector<std::unique_ptr<PlacedSuit>> &getPlayedCards() const;
  const std::vector<std::vector<bool>> &getPassRecord() const;
  const std::vector<std::vector<Card>> &getCoveredCards() const;

  // Internal helper for covering - NOW PUBLIC for Python Binding
  void coverCard(const Card &card);

  // State Initialization API for UI / Python binding
  void setFirstMovePerformed(bool val);
  void setCurrentPlayer(int playerNum);
  void setDealer(int playerNum);
  void setHand(int playerNum, const std::vector<Card> &cards);
  void setCoveredCards(int playerNum, const std::vector<Card> &cards);
  void setPlayedCardRange(int suit, int lowRank, int highRank);

  // AI Inference Helper
  void
  determinize(int observerPlayer,
              const std::vector<float> &beliefProbs = std::vector<float>());
  int getTurnCount() const;
  void setTurnCount(int count);

private:
  void performInitialDeal();
  void placeCard(const Card &card);
  void removeCardFromPlayersHand(int playerNumber, const Card &card);
  void checkIsValidPlayerIndex(int playerIndex) const;

  int numberOfPlayers;
  int currentPlayerNumber;
  int dealerNumber;
  bool firstMovePerformed;
  int turnCount;
  std::vector<Hand> hands;
  std::vector<std::vector<Card>> coveredCards; // Added for Cover Rule
  std::vector<std::unique_ptr<PlacedSuit>> playedCards;
  std::vector<std::vector<bool>> passRecord; // [playerIndex][suitIndex]

  int findPlayerWithCard(int suit, int rank) const;
};

#endif // SEVENSGAME_H
