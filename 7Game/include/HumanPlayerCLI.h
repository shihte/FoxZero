#ifndef HUMANPLAYERCLI_H
#define HUMANPLAYERCLI_H

#include "Player.h"
#include <optional>

class SevensGame;
class Hand;

class HumanPlayerCLI : public Player {
public:
  std::optional<Card> getMove(SevensGame &model,
                              int currentPlayerNumber) override;

private:
  int getSuit();
  int getRank();
};

#endif // HUMANPLAYERCLI_H
