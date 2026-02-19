/*
    SevensGame.java 1.0 2020/06/11
 */
package sevens.model.game;

import sevens.model.carddeck.Card;
import sevens.model.carddeck.Deck;
import sevens.model.carddeck.Hand;

import java.util.ArrayList;

/**
 * Represents a sevens card game.
 * <p>
 * Stores attributes related to the game such as the played cards and players
 * hands. Provides methods to interfaces to facilitate the playing of the game.
 *
 * @author Matthew Williams
 * @version 1.0     Initial Implementation
 */
public class SevensGame {

    // //////////////// //
    // Class variables. //
    // //////////////// //

    private final int numberOfPlayers;
    private Hand[] hands;
    private PlacedSuit[] playedCards; // Index 0=DIAMONDS; 1=CLUBS; 2=HEARTS; 3=SPADES
    private int currentPlayerNumber;

    // Card Constants:
    //  DIAMOND = 1
    //  CLUB = 2
    //  HEART = 3
    //  SPADE = 4

    // ///////////// //
    // Constructors. //
    // ///////////// //

    /**
     * Instantiates a new Sevens game.
     * <p>
     * Sets up the game with the default number of players (4)
     */
    public SevensGame() {
        this(4);
    }

    /**
     * Instantiates a new Sevens game.
     *
     * @param numberOfPlayers the number of players
     */
    public SevensGame(int numberOfPlayers) {
        this.numberOfPlayers = numberOfPlayers;
        this.currentPlayerNumber = 1;
        setupNewGame();
    }

    // ///////////////////// //
    // Read-only properties. //
    // ///////////////////// //

    /**
     * Gets the number of players.
     *
     * @return the number of players
     */
    public int getNumberOfPlayers() {
        return numberOfPlayers;
    }

    /**
     * Get all players hands
     *
     * @return an array containing all players hands
     */
    public Hand[] getHands() {
        return hands;
    }

    /**
     * Get each of the four placed card suits.
     *
     * @return an array containing the four placed card suits.
     */
    public PlacedSuit[] getPlayedCards() {
        return playedCards;
    }

    /**
     * Gets the current player number.
     *
     * @return the current player number
     */
    public int getCurrentPlayerNumber() {
        return currentPlayerNumber;
    }

    // //////// //
    // Methods. //
    // //////// //

    /**
     * Sets up a new game.
     * <p>
     * Resets each of the four placed suits and shuffles and deals the cards.
     */
    public void setupNewGame() {
        playedCards = new PlacedSuit[4];

        // instantiate each placed suit
        for (int i = 0; i < 4; i++) {
            playedCards[i] = new PlacedSuit(i+1);
        }

        // deal cards
        performInitialDeal();
    }

    /**
     * Performs an initial deal to all players.
     * <p>
     * Shuffles the card deck then deals all cards starting from the first
     * player and cycling around until all cards are dealt.
     */
    private void performInitialDeal() {
        Card dealtCard;
        int currentPlayerIndex = 0;

        // setup new deck and new hand states
        Deck deck = new Deck(false); // standard deck without jokers
        hands = new Hand[numberOfPlayers];

        // instantiate each hand
        for (int i = 0; i < numberOfPlayers; i++) {
            hands[i] = new Hand();
        }

        // deal all the cards in the deck
        while (deck.getNumberOfCardsRemaining() > 0) {
            dealtCard = deck.dealCard();
            hands[currentPlayerIndex].addCard(dealtCard);

            currentPlayerIndex++;
            // if we are on the last player, cycle back around to first player
            if (currentPlayerIndex == numberOfPlayers) {
                currentPlayerIndex = 0;
            }
        }
    }

    /**
     * Moves onto the next player.
     */
    public void nextPlayer() {
        this.currentPlayerNumber++;
        // if we are on the last player, cycle back around to first player
        if (this.currentPlayerNumber == (this.numberOfPlayers + 1)) {
            this.currentPlayerNumber = 1;
        }
    }

    /**
     * Determines if a given card can be played.
     *
     * @param cardToPlay the card to play
     * @return  true if the card can be played;
     *          false otherwise.
     */
    public boolean isValidMove(Card cardToPlay) {
        // get cards suit
        int cardsSuit = cardToPlay.getSuit();
        // get index of current suit
        int suitIndex = cardsSuit - 1;

        // lookup the suit in the playedCards array
        PlacedSuit placedSuit = playedCards[suitIndex];

        return placedSuit.canCardBePlaced(cardToPlay);
    }

    /**
     * Gets all valid moves for a given players hand.
     *
     * @param playerNumber the player number
     * @return the all valid moves the given player can make
     */
    public ArrayList<Card> getAllValidMoves(int playerNumber) {
        // create a new list to keep track of valid moves
        ArrayList<Card> validMoves = new ArrayList<>();

        // get player index and determine if the player number arg is valid
        int playerIndex = playerNumber - 1;
        checkIsValidPlayerIndex(playerIndex);
        // get the players hand
        Hand playersHand = hands[playerIndex];

        // iterate over each card in the players hand
        for (Card move : playersHand.getHand()) {
            // determine if can be played, if so add to our list of valid moves
            if (isValidMove(move)) {
                validMoves.add(move);
            }
        }

        return validMoves;
    }

    /**
     * Determines if a player index is within the acceptable range or not.
     *
     * @param playerIndex the player index to test
     * @throws IllegalArgumentException if the argument is an invalid player
     *                                  number - something is wrong with the
     *                                  implemented interface
     */
    private void checkIsValidPlayerIndex(int playerIndex)
        throws IllegalArgumentException {
        if ((playerIndex < 0) || (playerIndex >= numberOfPlayers)) {
            throw new IllegalArgumentException("Invalid Player Number");
        }
    }

    /**
     * Determines if a given player has won.
     *
     * @param playerNumber the player number to check
     * @return  true if the given player has no cards left (ie has won);
     *          false otherwise.
     */
    public boolean hasPlayerWon(int playerNumber) {
        // get player index and determine if the player number arg is valid
        int playerIndex = playerNumber - 1;
        checkIsValidPlayerIndex(playerIndex);

        // determine if the player has no cards left to play
        return hands[playerIndex].getCardCount() == 0;
    }

    /**
     * Gets a given players hand.
     *
     * @param playerNumber the player number
     * @return the players hand
     */
    public Hand getPlayersHand(int playerNumber) {
        // get player index and determine if the player number arg is valid
        int playerIndex = playerNumber - 1;
        checkIsValidPlayerIndex(playerIndex);

        return hands[playerIndex];
    }

    /**
     * Makes a given move.
     * <p>
     * Removes the given card from the players hand then places this on the
     * appropriate placed suit.
     * Does not perform checking, this must be done using the other methods
     * provided.
     *
     * @param cardToPlay the card to play
     */
    public void makeMove(Card cardToPlay) {
        // remove card from players hand
        removeCardFromPlayersHand(currentPlayerNumber, cardToPlay);
        // play the valid move
        placeCard(cardToPlay);
    }

    /**
     * Places a given card in the appropriate placed suit.
     * <p>
     * Does not perform checking, this must be done using the other methods
     * provided.
     *
     * @param cardToPlay the card to play
     */
    private void placeCard(Card cardToPlay) {
        // assumes is valid move

        // get cards suit and rank
        int cardsSuit = cardToPlay.getSuit();
        int cardsRank = cardToPlay.getRank();

        // get index of cards suit
        int suitIndex = cardsSuit - 1;

        // lookup the suit in the playedCards array
        PlacedSuit placedSuit = playedCards[suitIndex];

        // if card to play is a seven update both highest and lowest
        if (cardsRank == 7) {
            placedSuit.setLowestCard(cardToPlay);
            placedSuit.setHighestCard(cardToPlay);

            // if card to play is greater than a seven update highest
        } else if (cardsRank > 7) {
            placedSuit.setHighestCard(cardToPlay);

            // if card to play is less than than a seven update lowest
        } else { // cardsRank < 7
            placedSuit.setLowestCard(cardToPlay);
        }
    }

    /**
     * Removes a given card from the players hand.
     *
     * @param playerNumber the number of the player whose hand holds the card
     * @param cardToRemove the card to remove
     */
    private void removeCardFromPlayersHand(int playerNumber, Card cardToRemove) {
        // get player index and determine if the player number arg is valid
        int playerIndex = playerNumber - 1;
        checkIsValidPlayerIndex(playerIndex);

        hands[playerIndex].removeCard(cardToRemove);
    }

}