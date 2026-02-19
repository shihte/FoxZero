#include "ComputerPlayer.h"
#include "SevensGame.h"
#include <algorithm>
#include <climits>

std::optional<Card> ComputerPlayer::getMove(SevensGame &model,
                                            int currentPlayerNumber) {
  Hand &playersHand = model.getPlayersHand(currentPlayerNumber);
  std::vector<Card> validMoves = model.getAllValidMoves(currentPlayerNumber);

  if (validMoves.empty()) {
    return std::nullopt;
  } else if (validMoves.size() == 1) {
    return validMoves[0];
  } else {
    int currentMaxScore = INT_MIN;
    std::optional<Card> currentBestCard = std::nullopt;

    for (const auto &card : validMoves) {
      int score = scoreCard(model, playersHand, card);
      if (score > currentMaxScore) {
        currentMaxScore = score;
        currentBestCard = card;
      }
    }
    return currentBestCard;
  }
}

int ComputerPlayer::scoreCard(SevensGame &model, const Hand &playersHand,
                              const Card &assessedCard) {
  const auto &playedCards = model.getPlayedCards();
  const PlacedSuit &suitsState = *playedCards[assessedCard.getSuit() - 1];

  std::set<int> cardsOfSuitInHand;
  for (const auto &card : playersHand.getHand()) {
    if (card.getSuit() == assessedCard.getSuit()) {
      cardsOfSuitInHand.insert(card.getRank());
    }
  }

  if (assessedCard.getRank() == 7) {
    return lookBelowSeven(suitsState, cardsOfSuitInHand) +
           lookAboveSeven(suitsState, cardsOfSuitInHand);
  } else if (assessedCard.getRank() > 7) {
    return lookAboveSeven(suitsState, cardsOfSuitInHand);
  } else {
    return lookBelowSeven(suitsState, cardsOfSuitInHand);
  }
}

int ComputerPlayer::lookBelowSeven(const PlacedSuit &suitsState,
                                   const std::set<int> &cardsOfSuitInHand) {
  if (cardsOfSuitInHand.empty())
    return -6;

  int minCardRank = *cardsOfSuitInHand.begin();
  int cardsBelowSevenCount = 0;
  for (int rank : cardsOfSuitInHand) {
    if (rank < 7)
      cardsBelowSevenCount++;
  }

  if (cardsBelowSevenCount == 0) {
    return -6;
  } else {
    int lowestCardRank =
        suitsState.getLowestCard() ? suitsState.getLowestCard()->getRank() : 7;
    int score = lowestCardRank - minCardRank - 1;
    score -= (cardsBelowSevenCount - 1);
    score -= (minCardRank - Card::ACE);
    return score;
  }
}

int ComputerPlayer::lookAboveSeven(const PlacedSuit &suitsState,
                                   const std::set<int> &cardsOfSuitInHand) {
  if (cardsOfSuitInHand.empty())
    return -6;

  int maxCardRank = *cardsOfSuitInHand.rbegin();
  int cardsAboveSevenCount = 0;
  for (int rank : cardsOfSuitInHand) {
    if (rank > 7)
      cardsAboveSevenCount++;
  }

  if (cardsAboveSevenCount == 0) {
    return -6;
  } else {
    int highestCardRank = suitsState.getHighestCard()
                              ? suitsState.getHighestCard()->getRank()
                              : 7;
    int score = maxCardRank - highestCardRank - 1;
    score -= (cardsAboveSevenCount - 1);
    score -= (Card::KING - maxCardRank);
    return score;
  }
}
