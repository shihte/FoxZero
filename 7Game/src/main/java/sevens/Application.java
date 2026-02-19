/*
    Application.java 1.0.1 2020/06/18
 */
package sevens;

import sevens.views.cli.SevensGameCLI;

/**
 * The Application Class.
 *
 * @author Matthew Williams
 * @version 1.0     Initial Implementation
 * @version 1.0.1   Added File Header Comment
 */
public class Application {

    /**
     * The entry point of the application.
     *
     * @param args the arguments
     */
    public static void main(String[] args) {
        SevensGameCLI game = new SevensGameCLI();
        game.playGame();
    }

}