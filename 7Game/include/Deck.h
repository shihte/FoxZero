#ifndef DECK_H
#define DECK_H

#include "Card.h"
#include <vector>

class Deck {
public:
  Deck();
  Deck(bool addJokers);

  Card dealCard();
  void shuffle();
  int getNumberOfCardsRemaining() const;
  bool containsJokers() const;

private:
  std::vector<Card> deck;
  int numOfCardsDealt;
};

#endif // DECK_H
