from enum import IntEnum

class Suit(IntEnum):
    DIAMOND = 1
    CLUB = 2
    HEART = 3
    SPADE = 4
    JOKER = 5  # Not typically used in Sevens but kept for compatibility

    def __str__(self):
        return self.name.capitalize()

class Rank(IntEnum):
    ACE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    
    def __str__(self):
        if self == Rank.ACE: return "Ace"
        if self == Rank.JACK: return "Jack"
        if self == Rank.QUEEN: return "Queen"
        if self == Rank.KING: return "King"
        return str(self.value)

class Card:
    def __init__(self, suit: int, rank: int):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return f"{Rank(self.rank)} of {Suit(self.suit)}s"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.suit == other.suit and self.rank == other.rank
    
    def __hash__(self):
        return hash((self.suit, self.rank))

    def to_tensor_index(self):
        """
        Returns the (suit_idx, rank_idx) for the 4x13 tensor.
        Suit: Diamond=0, Club=1, Heart=2, Spade=3
        Rank: Ace=0 ... King=12
        """
        # Map Suit 1-4 to 0-3
        suit_idx = self.suit - 1
        # Map Rank 1-13 to 0-12
        rank_idx = self.rank - 1
        return suit_idx, rank_idx
