# Sevens Card Game

A Multi-Player Game based on the Popular [Domino](https://en.wikipedia.org/wiki/Domino_(card_game))/Sevens Card Game.

All cards are dealt to players - this may mean that some players have one more card than others.
The first player begins by playing a card of rank 7, if they are unable to play a card they skip their go.
Subsequent players must add a card one higher or one lower from any of the played suits. If they are unable to do this they skip their go.

The first player to have no cards left wins the game.

## C++ Build Instructions

```bash
mkdir build
cd build
cmake ..
make
./sevens_game
```

## Computer Player
A computer player has been implemented which scores cards which can be played in its hand and plays the optimal one to benefit itself.

### Method
For each valid move in the computer players hand, we calculate a score, this is done as follows:

1) If cards rank is 7, look above seven and below seven, sum these two generated scores
2) If cards rank is > 7, look above seven
3) If cards rank is < 7, look above seven

It then chooses to play the card with the highest score.

For the 'look' methods, we count how many cards we rely on other players playing between the current state of the suit on the table and our furthest away rank.

From this result we then subtract the number of cards other players can place after we've played our furthest away rank.

This is outlined in the example below.

### Example

Placed Cards:
```
Diamonds     6 to 8
Clubs        7 Only
Hearts       Suit not yet played
Spades       7 Only
```

Our Hand:
```
* 	5 of Diamonds
	Jack of Diamonds
	Queen of Clubs
	6 of Hearts
* 	7 of Hearts
	Ace of Spades
* 	6 of Spades
	Queen of Spades
```

The * represents a card that produces a valid move.

The algorithm scores each of these valid moves:
1) 5 of Diamonds - case 3.
We count the number of Diamonds cards we have below 7 = 1.
Our minimum card is the 5.
The current lowest played card is the 6.
We determine the number of cards between the lowest played card and our minimum card (6 - 5 - 1 = 0).
We subtract the number of cards we have between the lowest played card and our minimum card (0).
We then subtract the number of cards lower than our lowest card (Ace through Four, this is 4 cards, so 0 - 4).
Gives score of -4.

2) 7 of Hearts - case 1.
Looking at case 3 we have 1 card below 7 so below score is 0 - number of cards below 6 = 0 - 5 = -5.
Looking at case 2 we have 0 cards above 7, so above score is 0 - number of cards above 7 = 0 - 6 = -6.
Totalling this gives score of -11.

3) 6 of Spades - case 3.
We count the number of Spades cards we have below 7 = 2.
Our minimum card is the Ace (1).
The current lowest played card is the 7.
We determine the number of cards between the lowest played card and our minimum card (7 - 1 - 1 = 5).
We subtract the number of cards we have between the lowest played card and our minimum card (5 - (2 - 1) = 4).
We then subtract the number of cards lower than our lowest card (0).
Gives score of 4.

We therefore choose to play the 6 of Spades as it is most beneficial towards us given our hand.