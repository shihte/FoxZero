/*
    Player.java 1.0.1 2020/06/18
 */
package sevens.model.players;

import sevens.model.carddeck.Card;
import sevens.model.game.SevensGame;

/**
 * The interface Player.
 *
 * @author Matthew Williams
 * @version 1.0     Initial Implementation
 * @version 1.0.1   Added File Header Comment
 */
public interface Player {

    /**
     * Gets move.
     *
     * @param model               the model
     * @param currentPlayerNumber the current player number
     * @return  card of choice to play;
     *          null if no move can be made.
     */
    Card getMove(SevensGame model, int currentPlayerNumber);

}