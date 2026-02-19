#include "Card.h"
#include <stdexcept>

Card::Card() : suit(JOKER), rank(1) {}

Card::Card(int suit, int rank) {
  if (!isValidSuit(suit))
    throw std::invalid_argument("Invalid Card Suit");
  if (!isValidRank(suit, rank))
    throw std::invalid_argument("Invalid Card Rank");

  this->suit = suit;
  this->rank = rank;
}

int Card::getSuit() const { return suit; }

std::string Card::getSuitString() const {
  switch (suit) {
  case DIAMOND:
    return "方塊";
  case CLUB:
    return "梅花";
  case HEART:
    return "紅心";
  case SPADE:
    return "黑桃";
  default:
    return "鬼牌";
  }
}

int Card::getRank() const { return rank; }

std::string Card::getRankString() const {
  switch (rank) {
  case ACE:
    return "A";
  case JACK:
    return "J";
  case QUEEN:
    return "Q";
  case KING:
    return "K";
  default:
    return std::to_string(rank);
  }
}

std::string Card::toString() const {
  if (suit == JOKER)
    return "鬼牌 #" + std::to_string(rank);
  return getSuitString() + " " + getRankString();
}

bool Card::isValidSuit(int suit) {
  return (suit >= DIAMOND && suit <= SPADE) || (suit == JOKER);
}

bool Card::isValidRank(int suit, int rank) {
  return (suit == JOKER) || (rank >= ACE && rank <= KING);
}
