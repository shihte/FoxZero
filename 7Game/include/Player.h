#ifndef PLAYER_H
#define PLAYER_H

#include "Card.h"
#include <optional>

class SevensGame;

class Player {
public:
  virtual ~Player() = default;

  // Returns std::nullopt if the player skips their turn
  virtual std::optional<Card> getMove(SevensGame &model,
                                      int currentPlayerNumber) = 0;
};

#endif // PLAYER_H
