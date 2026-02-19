#include "HumanPlayerCLI.h"
#include "SevensGame.h"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <string>

std::optional<Card> HumanPlayerCLI::getMove(SevensGame &model,
                                            int currentPlayerNumber) {
  Hand &hand = model.getPlayersHand(currentPlayerNumber);
  bool canMakeMove = !model.getAllValidMoves(currentPlayerNumber).empty();

  if (!canMakeMove) {
    std::cout << "玩家 " << currentPlayerNumber
              << "，你沒有可以出的牌，跳過此回合。" << std::endl;
    return std::nullopt;
  }

  while (true) {
    int suit = -1;
    while (suit == -1) {
      suit = getSuit();
    }

    int rank = -1;
    while (rank == -1) {
      rank = getRank();
    }

    auto cardOpt = hand.getCard(suit, rank);
    if (cardOpt) {
      return *cardOpt;
    } else {
      std::cerr << "你的手牌中沒有這張牌。無效的牌。" << std::endl;
    }
  }
}

int HumanPlayerCLI::getSuit() {
  std::string suitInput;
  std::cout << "請輸入花色 (方塊, 梅花, 紅心, 黑桃):" << std::endl;
  if (!(std::cin >> suitInput))
    return -1;

  if (suitInput == "方塊" || suitInput == "DIAMONDS")
    return Card::DIAMOND;
  if (suitInput == "梅花" || suitInput == "CLUBS")
    return Card::CLUB;
  if (suitInput == "紅心" || suitInput == "HEARTS")
    return Card::HEART;
  if (suitInput == "黑桃" || suitInput == "SPADES")
    return Card::SPADE;

  std::cerr << "輸入的花色無效" << std::endl;
  return -1;
}

int HumanPlayerCLI::getRank() {
  std::string rankInput;
  std::cout << "請輸入點數 (A, 2-10, J, Q, K):" << std::endl;
  if (!(std::cin >> rankInput))
    return -1;

  std::transform(rankInput.begin(), rankInput.end(), rankInput.begin(),
                 ::toupper);

  if (rankInput == "A" || rankInput == "ACE")
    return Card::ACE;
  if (rankInput == "J" || rankInput == "JACK")
    return Card::JACK;
  if (rankInput == "Q" || rankInput == "QUEEN")
    return Card::QUEEN;
  if (rankInput == "K" || rankInput == "KING")
    return Card::KING;

  try {
    int rank = std::stoi(rankInput);
    if (rank > Card::ACE && rank < Card::JACK) {
      return rank;
    }
  } catch (...) {
    // Failed to parse or out of range
  }

  std::cerr << "輸入的點數無效" << std::endl;
  return -1;
}
