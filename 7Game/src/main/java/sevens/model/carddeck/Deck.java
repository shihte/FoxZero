package sevens.model.carddeck;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * The type Deck - represents a deck of playing cards.
 *
 * @author  Matthew Williams
 * @version 1.0.1
 * @since   2019-06-24
 */
public class Deck {

    private final Card[] deck;
    private int numOfCardsDealt;

    /**
     * Instantiates a new Deck.
     */
    public Deck() {
        this(false);
    }

    /**
     * Instantiates a new Deck.
     *
     * @param addJokers the add jokers
     */
    public Deck(boolean addJokers) {
        int cardCount = 0;

        if (addJokers) {
            deck = new Card[54];
            deck[52] = new Card(1, Card.JOKER);
            deck[53] = new Card(2, Card.JOKER);
        } else {
            deck = new Card[52];
        }

        for (int suit = 1; suit < 5; suit++)
            for (int rank = 1; rank <= Card.KING; rank++)
                deck[cardCount++] = new Card(suit, rank);

        shuffle();
    }

    /**
     * Deal card card.
     *
     * @return the card
     */
    public Card dealCard() {
        if (numOfCardsDealt == deck.length)
            throw new IllegalStateException("No Cards Remaining in Deck");
        return deck[numOfCardsDealt++];
    }

    /**
     * Shuffle.
     */
    public void shuffle() {
        numOfCardsDealt = 0;
        // Fisher-Yates shuffle:
        Random rand = ThreadLocalRandom.current();
        for (int i = deck.length - 1; i > 0; i--) {
            int index = rand.nextInt(i + 1);
            // Simple swap
            Card temp = deck[index];
            deck[index] = deck[i];
            deck[i] = temp;
        }
    }

    /**
     * Gets number of cards remaining.
     *
     * @return the number of cards remaining
     */
    public int getNumberOfCardsRemaining() {
        return deck.length - numOfCardsDealt;
    }

    /**
     * Contains jokers boolean.
     *
     * @return the boolean
     */
    public boolean containsJokers() {
        return (deck.length == 54);
    }

}
