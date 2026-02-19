package sevens.model.carddeck;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * The type Hand - represents a hand of playing cards.
 *
 * @author  Matthew Williams
 * @version 1.0
 * @since   2019-06-24
 */
public class Hand {

    private List<Card> hand;

    /**
     * Instantiates a new Hand.
     */
    public Hand() {
        hand = new ArrayList<>();
    }

    /**
     * Add card.
     *
     * @param card the card
     */
    public void addCard(Card card) {
        Objects.requireNonNull(card, "Invalid Card to Add");
        hand.add(card);
    }

    /**
     * Gets card.
     *
     * @param cardPosition the card position
     * @return the card
     */
    public Card getCard(int cardPosition) {
        checkValidPosition(cardPosition);
        return hand.get(cardPosition);
    }

    /**
     * Gets card.
     *
     * @param suit the suit of the card to retrieve
     * @param rank the rank of the card to retrieve
     * @return  the card if present in the users hand;
     *          null otherwise.
     */
    public Card getCard(int suit, int rank) {
        // check hand for card presence
        for (Card c : hand) {
            if ((c.getSuit() == suit) && (c.getRank() == rank)) {
                return c;
            }
        }
        // not present in hand
        return null;
    }

    /**
     * Gets the hand.
     *
     * @return the hand
     */
    public List<Card> getHand() {
        return hand;
    }

    /**
     * Remove card.
     *
     * @param card the card
     */
    public void removeCard(Card card) {
        hand.remove(card);
    }

    /**
     * Remove card.
     *
     * @param cardPosition the card position
     */
    public void removeCard(int cardPosition) {
        checkValidPosition(cardPosition);
        hand.remove(cardPosition);
    }

    /**
     * Sort hand by card rank.
     */
    public void sortByRank() {
        List<Card> newHand = new ArrayList<>();
        int minimalCardPosition;
        Card minimalCard;
        Card comparisonCard;
        while (hand.size() > 0) {
            minimalCardPosition = 0;
            minimalCard = hand.get(minimalCardPosition);
            for (int i = 1; i < hand.size(); i++) {
                comparisonCard = hand.get(i);
                if ((minimalCard.getRank() > comparisonCard.getRank()) ||
                        ((minimalCard.getRank() == comparisonCard.getRank()) && (minimalCard.getSuit() > comparisonCard.getSuit()))) {
                    // comparison card is lower in rank
                    minimalCardPosition = i;
                    minimalCard = comparisonCard;
                }
            }
            newHand.add(minimalCard);
            hand.remove(minimalCardPosition);
        }
        hand = newHand;
    }

    /**
     * Sort hand by card suit.
     */
    public void sortBySuit() {
        List<Card> newHand = new ArrayList<>();
        int minimalCardPosition;
        Card minimalCard;
        Card comparisonCard;
        while (hand.size() > 0) {
            minimalCardPosition = 0;
            minimalCard = hand.get(minimalCardPosition);
            for (int i = 1; i < hand.size(); i++) {
                comparisonCard = hand.get(i);
                if ((minimalCard.getSuit() > comparisonCard.getSuit()) ||
                        ((minimalCard.getSuit() == comparisonCard.getSuit()) && (minimalCard.getRank() > comparisonCard.getRank()))) {
                    // comparison card is lower in suit
                    minimalCardPosition = i;
                    minimalCard = comparisonCard;
                }
            }
            newHand.add(minimalCard);
            hand.remove(minimalCardPosition);
        }
        hand = newHand;
    }

    /**
     * Gets the number of cards in the hand.
     *
     * @return the card count in the hand
     */
    public int getCardCount() {
        return hand.size();
    }

    /**
     * Clear.
     */
    public void clear() {
        hand.clear();
    }

    private void checkValidPosition(int cardPosition) {
        if (cardPosition < 0 || cardPosition >= hand.size())
            throw new IllegalArgumentException("Invalid Position in Hand");
    }

}