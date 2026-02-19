#include "Hand.h"
#include <algorithm>
#include <stdexcept>

Hand::Hand() {}

void Hand::addCard(const Card &card) { hand.push_back(card); }

Card Hand::getCard(int cardPosition) const {
  checkValidPosition(cardPosition);
  return hand[cardPosition];
}

std::optional<Card> Hand::getCard(int suit, int rank) const {
  for (const auto &c : hand) {
    if (c.getSuit() == suit && c.getRank() == rank) {
      return c;
    }
  }
  return std::nullopt;
}

const std::vector<Card> &Hand::getHand() const { return hand; }

void Hand::removeCard(const Card &card) {
  auto it = std::find_if(hand.begin(), hand.end(), [&](const Card &c) {
    return c.getSuit() == card.getSuit() && c.getRank() == card.getRank();
  });
  if (it != hand.end()) {
    hand.erase(it);
  }
}

void Hand::removeCardAt(int cardPosition) {
  checkValidPosition(cardPosition);
  hand.erase(hand.begin() + cardPosition);
}

void Hand::sortByRank() {
  std::sort(hand.begin(), hand.end(), [](const Card &a, const Card &b) {
    if (a.getRank() != b.getRank()) {
      return a.getRank() < b.getRank();
    }
    return a.getSuit() < b.getSuit();
  });
}

void Hand::sortBySuit() {
  std::sort(hand.begin(), hand.end(), [](const Card &a, const Card &b) {
    if (a.getSuit() != b.getSuit()) {
      return a.getSuit() < b.getSuit();
    }
    return a.getRank() < b.getRank();
  });
}

int Hand::getCardCount() const { return (int)hand.size(); }

void Hand::clear() { hand.clear(); }

void Hand::checkValidPosition(int cardPosition) const {
  if (cardPosition < 0 || cardPosition >= (int)hand.size()) {
    throw std::out_of_range("Invalid Position in Hand");
  }
}
