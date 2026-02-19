/*
    ComputerPlayer.java 1.0 2020/06/18
 */
package sevens.model.players;

import sevens.model.carddeck.Card;
import sevens.model.carddeck.Hand;
import sevens.model.game.PlacedSuit;
import sevens.model.game.SevensGame;

import java.util.*;

/**
 * The Computer Player.
 * <p>
 * Scores each valid move in the current players hand and plays the card with
 * the best score.
 *
 * @author Matthew Williams
 * @version 1.0     Initial Implementation
 */
public class ComputerPlayer
        implements Player {

    /**
     * Gets the optimal move decided by the computer.
     *
     * @param model                 the sevens game model
     * @param currentPlayerNumber   the current player number
     * @return  card of choice to play;
     *          null if no move can be made.
     */
    public Card getMove(SevensGame model, int currentPlayerNumber) {
        Hand playersHand = model.getPlayersHand(currentPlayerNumber);
        ArrayList<Card> validMoves = model.getAllValidMoves(currentPlayerNumber);

        // check if there are any moves to play
        if (validMoves.isEmpty()) {
            return null; // skip go

        // if only one move, play that
        } else if (validMoves.size() == 1) {
            return validMoves.get(0);

        // determine which is the best move to make as >1 valid moves
        } else {
            // maintain knowledge of current best score
            int currentMaxScore = Integer.MIN_VALUE;
            Card currentBestCard = null;
            int score;

            // iterate over all valid moves determining a score for each
            for (Card card : validMoves) {
                score = scoreCard(model, playersHand, card);

                // determine if the card is the new best
                if (score > currentMaxScore) {
                    currentMaxScore = score;
                    currentBestCard = card;
                }
            }
            // return card with the best score
            return currentBestCard;
        }
    }

    /**
     * Scores a given card based on the state of the played cards.
     *
     * @param model         the sevens game model
     * @param playersHand   the players hand
     * @param assessedCard  the card to be scored/assessed
     * @return  the score the computer assigns the assessed card -
     *          the higher the better
     */
    private int scoreCard(SevensGame model, Hand playersHand, Card assessedCard) {
        // get the placed suit of the card from the model
        PlacedSuit suitsState = model.getPlayedCards()[assessedCard.getSuit() - 1];
        // get the rank of all cards of the placed suit from the players hand
        Set<Integer> cardsOfSuitInHand = new TreeSet<>();
        for (Card card : playersHand.getHand()) {
            if (card.getSuit() == assessedCard.getSuit()) {
                cardsOfSuitInHand.add(card.getRank());
            }
        }
        // determine where to look

        // if 7 then look above and below
        if (assessedCard.getRank() == 7) {
            return  lookBelowSeven(suitsState, cardsOfSuitInHand) +
                    lookAboveSeven(suitsState, cardsOfSuitInHand);

        // if >7 then look above
        } else if (assessedCard.getRank() > 7) {
            return lookAboveSeven(suitsState, cardsOfSuitInHand);

        // if <7 then look below
        } else {
            return lookBelowSeven(suitsState, cardsOfSuitInHand);
        }
    }

    /**
     * Scores a card by looking below seven for the given suit.
     *
     * @param suitsState        the present state of the cards suit
     * @param cardsOfSuitInHand a set of cards belonging to the given suit in
     *                          the players hand
     * @return the score the computer assigns to the suit below seven
     */
    private int lookBelowSeven(PlacedSuit suitsState, Set<Integer> cardsOfSuitInHand) {
        // get how many cards away the furthest card is
        int minCardRank = Collections.min(cardsOfSuitInHand);
        // count cards below 7 in hand
        int cardsBelowSevenCount = (int) cardsOfSuitInHand.parallelStream().filter(rank -> rank < 7).count();

        if (cardsBelowSevenCount == 0) {
            return -6; // we have no cards below 7, so all 6 belong to other people
        } else {
            // update score to be lowest current placed rank - minCardRank - 1
            int lowestCard = (suitsState.getLowestCard() == null) ?
                    7 : suitsState.getLowestCard().getRank();
            int score = lowestCard - minCardRank - 1;
            // subtract number of cards remaining to play below 7 minus 1 (-1 to disregard the
            // assessed card)
            //  if the player has more cards below the current lowest played then we don't depend
            //  on other players as much - prefer to play the ones we depend on them making the
            //  most moves to allow us to play
            score -= (cardsBelowSevenCount - 1);
            // subtract the number of cards other players can place after we've reached our lowest
            // card
            score -= minCardRank - Card.ACE;
            return score;
        }
    }

    /**
     * Scores a card by looking above seven for the given suit.
     *
     * @param suitsState        the present state of the cards suit
     * @param cardsOfSuitInHand a set of cards belonging to the given suit in
     *                          the players hand
     * @return the score the computer assigns to the suit above seven
     */
    private int lookAboveSeven(PlacedSuit suitsState, Set<Integer> cardsOfSuitInHand) {
        // get how many cards away the furthest card is
        int maxCardRank = Collections.max(cardsOfSuitInHand);
        // count cards above 7 in hand
        int cardsAboveSevenCount = (int) cardsOfSuitInHand.parallelStream().filter(rank -> rank > 7).count();

        if (cardsAboveSevenCount == 0) {
            return -6; // we have no cards above 7, so all 6 belong to other people
        } else {
            // update score to be maxCardRank - highest current placed rank - 1
            int highestCard = (suitsState.getHighestCard() == null) ?
                    7 : suitsState.getHighestCard().getRank();
            int score = maxCardRank - highestCard - 1;
            // subtract number of cards remaining to play above 7 minus 1 (-1 to disregard the
            // assessed card)
            //  if the player has more cards above the current highest played then we don't depend
            //  on other players as much - prefer to play the ones we depend on them making the
            //  most moves to allow us to play
            score -= (cardsAboveSevenCount - 1);
            // subtract the number of cards other players can place after we've reached our highest
            // card
            score -= Card.KING - maxCardRank;
            return score;
        }
    }

}