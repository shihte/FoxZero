#include "PlacedSuit.h"
#include <iomanip>
#include <sstream>

PlacedSuit::PlacedSuit(int suit) : suit(suit) {}

std::optional<Card> PlacedSuit::getLowestCard() const { return lowestCard; }

void PlacedSuit::setLowestCard(const Card &card) { lowestCard = card; }

std::optional<Card> PlacedSuit::getHighestCard() const { return highestCard; }

void PlacedSuit::setHighestCard(const Card &card) { highestCard = card; }

int PlacedSuit::getSuit() const { return suit; }

void PlacedSuit::setSuit(int suit) { this->suit = suit; }

std::string PlacedSuit::getSuitName() const {
  Card dummyCard(suit, 2);
  return dummyCard.getSuitString();
}

bool PlacedSuit::canCardBePlaced(const Card &cardToPlay) const {
  int rank = cardToPlay.getRank();
  if (lowestCard && highestCard) {
    if (rank > 7 && rank == highestCard->getRank() + 1) {
      return true;
    } else if (rank < 7 && rank == lowestCard->getRank() - 1) {
      return true;
    }
    return false;
  } else {
    return rank == 7;
  }
}

std::string PlacedSuit::toString() const {
  std::ostringstream oss;
  oss << std::left << std::setw(13) << getSuitName();

  if (!lowestCard || !highestCard) {
    oss << "尚未出牌";
  } else {
    int lowVal = lowestCard->getRank();
    int highVal = highestCard->getRank();

    if (lowVal == 7 && highVal == 7) {
      oss << "只有 7";
    } else {
      oss << lowestCard->getRankString() << " 到 "
          << highestCard->getRankString();
    }
  }
  return oss.str();
}
