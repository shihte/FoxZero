package sevens.model.carddeck;

/**
 * The type Card. Represents a playing card from a standard deck - including
 * the representation of jokers. Jokers can have a value - say for the case
 * where there are multiple jokers.
 *
 * @author  Matthew Williams
 * @version 1.0
 * @since   2019-06-24
 */
public class Card {

    // define the four suits and joker card type
    public final static int DIAMOND = 1;
    public final static int CLUB = 2;
    public final static int HEART = 3;
    public final static int SPADE = 4;
    public final static int JOKER = 5;

    // define the four non numeric cards
    public final static int ACE = 1;
    public final static int JACK = 11;
    public final static int QUEEN = 12;
    public final static int KING = 13;

    private final int suit;
    private final int rank;

    /**
     * Instantiates a new Card.
     */
    public Card() {
        suit = JOKER;
        rank = 1;
    }

    /**
     * Instantiates a new Card.
     *
     * @param suit the suit
     * @param rank the rank
     */
    public Card(int suit, int rank) {
        if (!isValidSuit(suit))
            throw new IllegalArgumentException("Invalid Card Suit");
        if (!isValidRank(suit, rank))
            throw new IllegalArgumentException("Invalid Card Rank");

        // both valid, add to card
        this.suit = suit;
        this.rank = rank;
    }

    /**
     * Gets suit.
     *
     * @return the suit
     */
    public int getSuit() {
        return suit;
    }

    /**
     * Gets suit as a string.
     *
     * @return the suit as a string
     */
    public String getSuitString() {
        switch (suit) {
            case DIAMOND:   return "Diamonds";
            case CLUB:      return "Clubs";
            case HEART:     return "Hearts";
            case SPADE:     return "Spades";
            default:        return "Joker";
        }
    }

    /**
     * Gets rank.
     *
     * @return the rank
     */
    public int getRank() {
        return rank;
    }

    /**
     * Gets rank as a string.
     *
     * @return the rank as a string
     */
    public String getRankString() {
        switch (rank) {
            case ACE:       return "Ace";
            case JACK:      return "Jack";
            case QUEEN:     return "Queen";
            case KING:      return "King";
            default:		return Integer.toString(rank);
        }
    }

    /**
     * Returns a string representation of the card.
     *
     * @return the card as a string
     */
    public String toString() {
        if (suit == JOKER)
            return "Joker #" + rank;
        return getRankString() + " of " + getSuitString();
    }

    private static boolean isValidSuit(int suit) {
        return (suit == DIAMOND) || (suit == CLUB) || (suit == HEART) || (suit == SPADE) || (suit == JOKER);
    }

    private static boolean isValidRank(int suit, int rank) {
        return (suit == JOKER) || (rank >= ACE && rank <= KING);
    }

}