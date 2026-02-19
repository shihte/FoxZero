#ifndef HAND_H
#define HAND_H

#include "Card.h"
#include <optional>
#include <vector>

class Hand {
public:
  Hand();

  void addCard(const Card &card);
  Card getCard(int cardPosition) const;
  std::optional<Card> getCard(int suit, int rank) const;
  const std::vector<Card> &getHand() const;

  void removeCard(const Card &card);
  void removeCardAt(int cardPosition);

  void sortByRank();
  void sortBySuit();

  int getCardCount() const;
  void clear();

private:
  std::vector<Card> hand;
  void checkValidPosition(int cardPosition) const;
};

#endif // HAND_H
