#ifndef PLACEDSUIT_H
#define PLACEDSUIT_H

#include "Card.h"
#include <optional>
#include <string>

class PlacedSuit {
public:
  PlacedSuit(int suit);

  std::optional<Card> getLowestCard() const;
  void setLowestCard(const Card &card);

  std::optional<Card> getHighestCard() const;
  void setHighestCard(const Card &card);

  int getSuit() const;
  void setSuit(int suit);

  std::string getSuitName() const;
  bool canCardBePlaced(const Card &card) const;
  std::string toString() const;

private:
  std::optional<Card> lowestCard;
  std::optional<Card> highestCard;
  int suit;
};

#endif // PLACEDSUIT_H
