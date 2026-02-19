/*
    PlacedSuit.java 1.0 2020/06/11
 */
package sevens.model.game;

import sevens.model.carddeck.Card;

/**
 * Represents a placed suit.
 * <p>
 * Stores attributes related to a suit that has been placed down - the highest
 * card, the lowest card and the suits identifier.
 *
 * @author Matthew Williams
 * @version 1.0     Initial Implementation
 */
public class PlacedSuit {

    // //////////////// //
    // Class variables. //
    // //////////////// //

    private Card lowestCard;
    private Card highestCard;
    private int suit;

    // ///////////// //
    // Constructors. //
    // ///////////// //

    /**
     * Instantiates a new Placed suit.
     */
    public PlacedSuit(int suit) {
        this.lowestCard = null;
        this.highestCard = null;
        this.suit = suit;
    }

    // ////////////////////// //
    // Read/Write properties. //
    // ////////////////////// //

    /**
     * Gets lowest card.
     *
     * @return the lowest card
     */
    public Card getLowestCard() {
        return this.lowestCard;
    }

    /**
     * Sets lowest card.
     *
     * @param lowestCard the new lowest card
     */
    public void setLowestCard(Card lowestCard) {
        this.lowestCard = lowestCard;
    }

    /**
     * Gets highest card.
     *
     * @return the highest card
     */
    public Card getHighestCard() {
        return this.highestCard;
    }

    /**
     * Sets highest card.
     *
     * @param highestCard the new highest card
     */
    public void setHighestCard(Card highestCard) {
        this.highestCard = highestCard;
    }

    /**
     * Gets suit.
     *
     * @return the suit
     */
    public int getSuit() {
        return this.suit;
    }

    /**
     * Sets suit.
     *
     * @param suit the new suit
     */
    public void setSuit(int suit) {
        this.suit = suit;
    }

    // //////// //
    // Methods. //
    // //////// //

    /**
     * Gets suit name.
     *
     * @return the suit name
     */
    public String getSuitName() {
        // create a dummy card
        Card dummyCard = new Card(this.suit, 2); // random rank
        return dummyCard.getSuitString();
    }

    /**
     * Determines if a card can be placed down.
     *
     * @param cardToPlay the card to play
     * @return  true if the card can be played on the current placed suit;
     *          false otherwise.
     */
    public boolean canCardBePlaced(Card cardToPlay) {
        // get cards rank
        int cardsRank = cardToPlay.getRank();

        // if at least one card placed (including the 7)
        if ((lowestCard != null) && (highestCard != null)) {
            // if >7 then valid to play if card value is 1 more than high value
            if ((cardsRank > 7) && (cardsRank == highestCard.getRank() + 1)) {
                return true;

            // if <7 then valid to play if card value is one less than low value
            } else {
                return (cardsRank < 7) && (cardsRank == lowestCard.getRank() - 1);
            }

        // if 7 then valid to play if not already there
        } else {
            return cardsRank == 7;
        }
    }

    /**
     * Returns a string representation of the placed suit.
     *
     * @return a string representation of the placed suit
     */
    @Override
    public String toString() {
        String suitName = String.format("%-13s", getSuitName()); // pad out and left align

        // if no cards placed
        if ((lowestCard == null) || (highestCard == null)) {
            return suitName + "Suit not yet played";
        }

        int lowVal = lowestCard.getRank();
        int highVal = highestCard.getRank();

        // if 7 is the only placed card
        if ((lowVal == 7) && (highVal == 7)) {
            return suitName + "7 Only";

        // if at least two cards placed (including the 7)
        } else {
            return suitName + lowestCard.getRankString() + " to " + highestCard.getRankString();
        }
    }

}