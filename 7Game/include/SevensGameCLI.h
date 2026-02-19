#ifndef SEVENSGAMECLI_H
#define SEVENSGAMECLI_H

#include "Player.h"
#include "SevensGame.h"
#include <memory>

class SevensGameCLI {
public:
  SevensGameCLI();
  void playGame();

private:
  int getNumberOfPlayersAsInput();
  int getNumberOfComputerPlayersAsInput();
  void displayHand(int playerNumber);
  void displayGameState();
  bool hasPlayerWon();
  bool isValidMove(const Card &card);

  std::unique_ptr<SevensGame> model;
  int numberOfComputerPlayers;
};

#endif // SEVENSGAMECLI_H
