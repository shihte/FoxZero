#include "SevensGameCLI.h"
#include "ComputerPlayer.h"
#include "HumanPlayerCLI.h"
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>

SevensGameCLI::SevensGameCLI() {
  int totalPlayers = getNumberOfPlayersAsInput();
  model = std::make_unique<SevensGame>(totalPlayers);
  numberOfComputerPlayers = getNumberOfComputerPlayersAsInput();
}

void SevensGameCLI::playGame() {
  bool gameWon = false;

  while (!gameWon) {
    int currentPlayerNumber = model->getCurrentPlayerNumber();
    displayGameState();

    std::unique_ptr<Player> currentPlayer;
    if (currentPlayerNumber >
        (model->getNumberOfPlayers() - numberOfComputerPlayers)) {
      currentPlayer = std::make_unique<ComputerPlayer>();
    } else {
      currentPlayer = std::make_unique<HumanPlayerCLI>();
      displayHand(currentPlayerNumber);
    }

    std::optional<Card> cardToPlay;
    while (true) {
      cardToPlay = currentPlayer->getMove(*model, currentPlayerNumber);
      if (cardToPlay) {
        if (isValidMove(*cardToPlay)) {
          break;
        } else {
          std::cerr << cardToPlay->toString() << " 無法出牌。無效的出牌。"
                    << std::endl;
        }
      } else {
        // Skips turn
        break;
      }
    }

    if (cardToPlay) {
      std::cout << "正在出牌：" << cardToPlay->toString() << std::endl;
      model->makeMove(*cardToPlay);
      gameWon = hasPlayerWon();
    } else {
      model->recordPass(currentPlayerNumber);
    }

    if (!gameWon) {
      model->nextPlayer();
    }
  }

  displayGameState();

  auto result = model->calculateFinalRewards();
  std::cout << "### 結算結果 ###" << std::endl;
  std::cout << "莊家是玩家 #" << model->getDealerNumber() << std::endl;
  for (int i = 0; i < model->getNumberOfPlayers(); ++i) {
    std::cout << "玩家 #" << (i + 1) << ": 罰分 " << result.rawPenalties[i]
              << " | 獎勵值 " << std::fixed << std::setprecision(4)
              << result.normalizedRewards[i] << std::endl;
  }

  std::cout << "遊戲結束" << std::endl;
}

int SevensGameCLI::getNumberOfPlayersAsInput() {
  int num;
  while (true) {
    std::cout << "請輸入玩家人數：" << std::endl;
    if (std::cin >> num && num >= 2 && num <= 52)
      break;
    std::cerr << "玩家人數必須介於 2 到 52 之間。請再試一次：" << std::endl;
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  return num;
}

int SevensGameCLI::getNumberOfComputerPlayersAsInput() {
  int total = model->getNumberOfPlayers();
  int num;
  while (true) {
    std::cout << "請輸入電腦玩家的人數：" << std::endl;
    if (std::cin >> num && num >= 0 && num <= total)
      break;
    std::cerr << "電腦玩家人數必須介於 0 到 " << total << " 之間。請再試一次："
              << std::endl;
    std::cin.clear();
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  }
  return num;
}

void SevensGameCLI::displayHand(int playerNumber) {
  Hand &hand = model->getPlayersHand(playerNumber);
  hand.sortBySuit();

  std::cout << "玩家 #" << playerNumber << " 的手牌：" << std::endl;
  for (const auto &card : hand.getHand()) {
    if (model->isValidMove(card)) {
      std::cout << "* \t" << card.toString() << std::endl;
    } else {
      std::cout << "\t" << card.toString() << std::endl;
    }
  }
}

void SevensGameCLI::displayGameState() {
  int current = model->getCurrentPlayerNumber();
  const auto &hands = model->getHands();

  std::cout << "\n### 目前遊戲狀態 ###" << std::endl;
  std::cout << "目前的玩家是：#" << current << std::endl;
  if (model->isFirstMove()) {
    std::cout << "【規則】持有 黑桃 7 的玩家必須首發出牌！" << std::endl;
  }
  std::cout << "桌面已出的牌：" << std::endl;
  for (const auto &suit : model->getPlayedCards()) {
    std::cout << "\t" << suit->toString() << std::endl;
  }
  std::cout << "玩家手牌餘數：" << std::endl;
  for (size_t i = 0; i < hands.size(); ++i) {
    std::cout << "\t玩家 " << (i + 1) << " 剩下 " << hands[i].getCardCount()
              << " 張牌" << std::endl;
  }
  std::cout << "######\n" << std::endl;
}

bool SevensGameCLI::hasPlayerWon() {
  int current = model->getCurrentPlayerNumber();
  if (model->hasPlayerWon(current)) {
    std::cout << "玩家 #" << current << " 贏了！" << std::endl;
    return true;
  }
  return false;
}

bool SevensGameCLI::isValidMove(const Card &card) {
  if (!model->isValidMove(card)) {
    std::cerr << "無效的出牌！" << std::endl;
    return false;
  }
  return true;
}
