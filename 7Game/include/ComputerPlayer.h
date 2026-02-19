#ifndef COMPUTERPLAYER_H
#define COMPUTERPLAYER_H

#include "Player.h"
#include <optional>
#include <set>
#include <string>

class SevensGame;
class Hand;
class PlacedSuit;

class ComputerPlayer : public Player {
public:
  std::optional<Card> getMove(SevensGame &model,
                              int currentPlayerNumber) override;

private:
  int scoreCard(SevensGame &model, const Hand &playersHand,
                const Card &assessedCard);
  int lookBelowSeven(const PlacedSuit &suitsState,
                     const std::set<int> &cardsOfSuitInHand);
  int lookAboveSeven(const PlacedSuit &suitsState,
                     const std::set<int> &cardsOfSuitInHand);
};

#endif // COMPUTERPLAYER_H
