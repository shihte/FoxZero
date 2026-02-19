#ifndef CARD_H
#define CARD_H

#include <string>

class Card {
public:
    enum Suit {
        DIAMOND = 1,
        CLUB = 2,
        HEART = 3,
        SPADE = 4,
        JOKER = 5
    };

    enum Rank {
        ACE = 1,
        JACK = 11,
        QUEEN = 12,
        KING = 13
    };

    Card(); // Default is Joker
    Card(int suit, int rank);

    int getSuit() const;
    std::string getSuitString() const;
    int getRank() const;
    std::string getRankString() const;
    std::string toString() const;

    static bool isValidSuit(int suit);
    static bool isValidRank(int suit, int rank);

private:
    int suit;
    int rank;
};

#endif // CARD_H
