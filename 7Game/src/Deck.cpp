#include "Deck.h"
#include <algorithm>
#include <chrono>
#include <random>
#include <stdexcept>

Deck::Deck() : Deck(false) {}

Deck::Deck(bool addJokers) : numOfCardsDealt(0) {
  if (addJokers) {
    deck.push_back(Card(Card::JOKER, 1));
    deck.push_back(Card(Card::JOKER, 2));
  }

  for (int suit = 1; suit <= 4; ++suit) {
    for (int rank = 1; rank <= Card::KING; ++rank) {
      deck.push_back(Card(suit, rank));
    }
  }

  shuffle();
}

Card Deck::dealCard() {
  if (numOfCardsDealt == (int)deck.size())
    throw std::runtime_error("No Cards Remaining in Deck");
  return deck[numOfCardsDealt++];
}

void Deck::shuffle() {
  numOfCardsDealt = 0;
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::shuffle(deck.begin(), deck.end(), std::default_random_engine(seed));
}

int Deck::getNumberOfCardsRemaining() const {
  return (int)deck.size() - numOfCardsDealt;
}

bool Deck::containsJokers() const { return deck.size() > 52; }
