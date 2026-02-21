import random
from typing import List, Optional, Tuple, Dict
from foxzero.card import Card, Suit, Rank

class Hand:
    def __init__(self):
        self.cards: List[Card] = []

    def add_card(self, card: Card):
        self.cards.append(card)

    def remove_card(self, card: Card):
        if card in self.cards:
            self.cards.remove(card)
    
    def get_card_count(self) -> int:
        return len(self.cards)
    
    def has_card(self, suit: int, rank: int) -> bool:
        return any(c.suit == suit and c.rank == rank for c in self.cards)

    def sort_by_rank(self):
        self.cards.sort(key=lambda c: (c.rank, c.suit))

    def sort_by_suit(self):
        self.cards.sort(key=lambda c: (c.suit, c.rank))

class Deck:
    def __init__(self, add_jokers: bool = False):
        self.cards: List[Card] = []
        for suit in [Suit.DIAMOND, Suit.CLUB, Suit.HEART, Suit.SPADE]:
            for rank in range(1, 14):
                self.cards.append(Card(suit, rank))
        
        if add_jokers:
            self.cards.append(Card(Suit.JOKER, 1))
            self.cards.append(Card(Suit.JOKER, 2))
        
        self.shuffle()
        self.dealt_index = 0

    def shuffle(self):
        random.shuffle(self.cards)
        self.dealt_index = 0

    def deal_card(self) -> Card:
        if self.dealt_index >= len(self.cards):
            raise IndexError("No cards remaining in deck")
        card = self.cards[self.dealt_index]
        self.dealt_index += 1
        return card
    
    def remaining(self) -> int:
        return len(self.cards) - self.dealt_index

class PlacedSuit:
    def __init__(self, suit: int):
        self.suit = suit
        self.lowest_card: Optional[Card] = None
        self.highest_card: Optional[Card] = None

    def can_card_be_placed(self, card: Card) -> bool:
        if card.suit != self.suit:
            return False
        
        if self.lowest_card is None or self.highest_card is None:
            return card.rank == 7
        
        if card.rank == self.highest_card.rank + 1:
            return True
        
        if card.rank == self.lowest_card.rank - 1:
            return True
            
        return False

    def place_card(self, card: Card):
        if card.rank == 7:
            self.lowest_card = card
            self.highest_card = card
        elif self.highest_card and card.rank > 7:
            self.highest_card = card
        elif self.lowest_card and card.rank < 7:
            self.lowest_card = card

class SevensGame:
    def __init__(self, num_players: int = 4):
        self.num_players = num_players
        self.hands: List[Hand] = [Hand() for _ in range(num_players)]
        self.played_cards: List[PlacedSuit] = [PlacedSuit(s) for s in [1, 2, 3, 4]]
        self.covered_cards: List[List[Card]] = [[] for _ in range(num_players)]
        # pass_record[player_idx][suit_idx]
        self.pass_record: List[List[bool]] = [[False]*4 for _ in range(num_players)]
        self.forbidden_cards: List[set] = [set() for _ in range(num_players)] # Set of Card objects (or tuples)

        self.dealer_number = 1
        self.current_player_number = 1
        self.first_move_performed = False
        self.turn_count = 0
        
        self.setup_new_game()

    def setup_new_game(self):
        # Reset state
        self.hands = [Hand() for _ in range(self.num_players)]
        self.covered_cards = [[] for _ in range(self.num_players)]
        self.played_cards = [PlacedSuit(s) for s in [1, 2, 3, 4]]
        self.pass_record = [[False]*4 for _ in range(self.num_players)]
        self.forbidden_cards = [set() for _ in range(self.num_players)]
        self.first_move_performed = False
        self.turn_count = 0
        
        # Deal
        self.perform_initial_deal()
        
        # Find dealer (Spade 7 holder)
        self.dealer_number = self.find_player_with_card(Suit.SPADE, 7)
        self.current_player_number = self.dealer_number

    def perform_initial_deal(self):
        deck = Deck()
        current_idx = 0
        while deck.remaining() > 0:
            self.hands[current_idx].add_card(deck.deal_card())
            current_idx = (current_idx + 1) % self.num_players
    
    def find_player_with_card(self, suit: int, rank: int) -> int:
        for i, hand in enumerate(self.hands):
            if hand.has_card(suit, rank):
                return i + 1
        return 1

    def get_playable_cards_on_board(self) -> List[Card]:
        """Returns all cards that would be valid moves if held by someone."""
        candidates = []
        for suit in [Suit.SPADE, Suit.HEART, Suit.DIAMOND, Suit.CLUB]:
            ps = self.played_cards[suit - 1]
            if ps.lowest_card is None:
                # 7 is playable
                candidates.append(Card(suit, 7))
            else:
                # Low - 1
                if ps.lowest_card.rank > 1:
                    candidates.append(Card(suit, ps.lowest_card.rank - 1))
                # High + 1
                if ps.highest_card.rank < 13:
                    candidates.append(Card(suit, ps.highest_card.rank + 1))
        return candidates

    def determinize(self, observer_player: int):
        """
        Randomizes properties of hidden hands AND covered cards, respecting known constraints.
        Constraints:
        1. Observer's hand is fixed.
        2. Players cannot hold cards they are 'forbidden' to have (because they passed/covered previously).
        3. Covered cards are also unknown to observer (shuffled back into pool).
        """
        import random
        
        # 0. Identify knowns
        observer_idx = observer_player - 1
        opponent_indices = [i for i in range(self.num_players) if i != observer_idx]
        
        # 1. Collect all unknown cards & current counts
        unknown_cards = []
        hand_counts = {}
        covered_counts = {}
        
        for i in opponent_indices:
            hand_counts[i] = len(self.hands[i].cards)
            covered_counts[i] = len(self.covered_cards[i])
            
            unknown_cards.extend(self.hands[i].cards)
            unknown_cards.extend(self.covered_cards[i])
            
            self.hands[i].cards = [] # Clear hand
            self.covered_cards[i] = [] # Clear covered
            
        # 2. Constraint Satisfaction Shuffle
        # Tries to shuffle until valid assignment found.
        # Fallback to random if too hard (e.g. constraints conflict due to bad tracking or deep search).
        
        max_attempts = 5
        success = False
        
        for attempt in range(max_attempts):
            random.shuffle(unknown_cards)
            
            # Check if this permutation is valid
            current_idx = 0
            possible = True
            
            temp_hands = {}
            temp_covered = {}
            
            for i in opponent_indices:
                # 1. Assign Covered (No constraints usually, or complex history ones we skip)
                num_cov = covered_counts[i]
                assigned_cov = unknown_cards[current_idx : current_idx + num_cov]
                current_idx += num_cov
                temp_covered[i] = assigned_cov
                
                # 2. Assign Hand (Check forbidden)
                num_hand = hand_counts[i]
                assigned_hand = unknown_cards[current_idx : current_idx + num_hand]
                current_idx += num_hand
                
                # Check constraints for HAND
                is_forbidden = False
                for c in assigned_hand:
                    for fc in self.forbidden_cards[i]:
                        if c.suit == fc.suit and c.rank == fc.rank:
                            is_forbidden = True
                            break
                    if is_forbidden: break
                    
                if is_forbidden:
                    possible = False
                    break
                    
                temp_hands[i] = assigned_hand
                
            if possible:
                # Apply
                for i in opponent_indices:
                    self.hands[i].cards = temp_hands[i]
                    self.hands[i].sort_by_suit()
                    self.covered_cards[i] = temp_covered[i]
                success = True
                break
                
        if not success:
            # Fallback: Just deal randomly (ignore constraints to keep game running)
            current_idx = 0
            for i in opponent_indices:
                # Covered
                num_cov = covered_counts[i]
                self.covered_cards[i] = unknown_cards[current_idx : current_idx + num_cov]
                current_idx += num_cov
                
                # Hand
                num_hand = hand_counts[i]
                self.hands[i].cards = unknown_cards[current_idx : current_idx + num_hand]
                current_idx += num_hand
                self.hands[i].sort_by_suit()

    def is_valid_move(self, card: Card) -> bool:
        if not self.first_move_performed:
            return card.suit == Suit.SPADE and card.rank == 7
        
        suit_idx = card.suit - 1
        return self.played_cards[suit_idx].can_card_be_placed(card)

    def get_all_valid_moves(self, player_number: int) -> List[Card]:
        hand = self.hands[player_number - 1]
        valid_on_board = [c for c in hand.cards if self.is_valid_move(c)]
        
        # Rule: If you can play, you MUST play.
        if valid_on_board:
            return valid_on_board
            
        # Rule: If you cannot play any card, you MUST cover one.
        # Any card in hand is a valid action.
        return list(hand.cards)

    def make_move(self, card: Card):
        player_idx = self.current_player_number - 1
        if card not in self.hands[player_idx].cards:
             raise ValueError(f"Player {self.current_player_number} does not have card {card}")
        
        # Check if this is a valid board move
        is_board_move = self.is_valid_move(card)
        
        # Enforce "Must Play" Rule
        if not is_board_move:
             # Check if player HAD any valid move
             hand = self.hands[player_idx]
             can_play_any = any(self.is_valid_move(c) for c in hand.cards)
             if can_play_any:
                 raise ValueError(f"Player {self.current_player_number} must play a valid card if able, cannot cover {card}")
             
             # If we are here, it's a valid COVER action.
             # KEY LOGIC: If they cover, they MUST NOT have any card that is currently playable.
             # We add all currently playable cards to their forbidden set.
             playable_now = self.get_playable_cards_on_board()
             for c_p in playable_now:
                 # Copy card to store in forbidden set
                 self.forbidden_cards[player_idx].add(Card(c_p.suit, c_p.rank))

             # Remove card from hand
             self.hands[player_idx].remove_card(card)
             self.covered_cards[player_idx].append(card)
             
        else:
             # Standard Move
             self.hands[player_idx].remove_card(card)
             suit_idx = card.suit - 1
             self.played_cards[suit_idx].place_card(card)
             self.first_move_performed = True
 
        self.turn_count += 1

    def record_pass(self, player_number: int):
        # Deprecated: Sevens doesn't have "Pass". You must cover.
        # But keeping for compatibility if needed, or raise error.
        raise NotImplementedError("Pass is not allowed in Sevens. You must cover a card.")

    def next_player(self):
        self.current_player_number = (self.current_player_number % self.num_players) + 1

    def has_player_won(self, player_number: int) -> bool:
        return self.hands[player_number - 1].get_card_count() == 0

    def is_game_over(self) -> bool:
        if self.turn_count >= 52:
            return True
        for p in range(1, self.num_players + 1):
            if self.has_player_won(p):
                return True
        return False

    def calculate_final_rewards(self) -> List[float]:
        """
        Calculates normalized rewards [-1, 1].
        Rules:
        - Base Penalty = Hand + Covered.
        - Doom: Penalty > 50 -> x2 Personal Penalty.
        - Dealer Lose: x2 Personal Penalty.
        - Dealer Win: x2 Winnings (Everyone else pays x2).
        - Finisher Win: x2 Winnings (Everyone else pays x2).
        """
        
        # 1. Base Penalties
        base_penalties = []
        for i in range(self.num_players):
            p = sum(c.rank for c in self.hands[i].cards) + sum(c.rank for c in self.covered_cards[i])
            base_penalties.append(p)
            
        # 2. Determine Winner (Lowest Base Penalty)
        # Use simple argmin. If tie, first player wins (or split? keep simple for now).
        min_p = min(base_penalties)
        winner_idx = base_penalties.index(min_p)
        
        # 3. Calculate Personal Adjusted Losses for Losers
        personal_losses = []
        for i in range(self.num_players):
            if i == winner_idx:
                personal_losses.append(0.0) # Winner pays nothing initially
                continue
                
            loss = float(base_penalties[i])
            
            # Doom Rule
            if base_penalties[i] > 50:
                loss *= 2.0
                
            # Dealer Lose Rule
            if (i + 1) == self.dealer_number:
                loss *= 2.0
                
            personal_losses.append(loss)
            
        # 4. Apply Winner Multipliers
        # Scaling factor S applies to the payouts from losers to winner.
        winner_multiplier = 1.0
        
        # Dealer Win Rule
        if (winner_idx + 1) == self.dealer_number:
            winner_multiplier *= 2.0
            
        # Finisher Win Rule
        # Player is "Finisher" if they have 0 cards in hand.
        if self.hands[winner_idx].get_card_count() == 0:
            winner_multiplier *= 2.0
            
        # 5. Finalize Rewards
        rewards = []
        total_pot = 0.0
        
        for i in range(self.num_players):
            if i == winner_idx:
                rewards.append(0.0) # Placeholder
            else:
                # Loser pays: PersonalLoss * WinnerMultiplier
                pay = personal_losses[i] * winner_multiplier
                rewards.append(-pay)
                total_pot += pay
                
        # Winner takes the pot
        rewards[winner_idx] = total_pot
        
        # 6. Normalize
        # Max theoretical: 
        # Base ~170. Doom(x2) * Dealer(x2) * WinnerAsDealer(x2) * WinnerAsFinisher(x2) = 16x multiplier?
        # That's huge. 170 * 16 = 2720.
        SCALE_FACTOR = 3000.0
        normalized_rewards = [r / SCALE_FACTOR for r in rewards]
        
        return normalized_rewards

    def get_state_tensor(self, player_number: int):
        """
        Returns 11x4x13 tensor for the given player.
        """
        # We need numpy or just use list of lists of lists
        # Let's try to return a nested list structure for now since numpy might not be available
        # But for training we definitely need numpy/torch.
        # Assuming numpy is available for simplicity of indices basically
        
        import numpy as np # Import locally to allow failure if not present (though we checked)
        
        state = np.zeros((11, 4, 13), dtype=np.float32)
        player_idx = player_number - 1
        
        # Ch 0: My Hand
        for c in self.hands[player_idx].cards:
            s_idx, r_idx = c.to_tensor_index()
            state[0, s_idx, r_idx] = 1.0
            
        # Ch 1: Board State
        # Mark all played cards as 1?
        # A card is played if it is between low and high (inclusive)
        for s_idx in range(4):
            ps = self.played_cards[s_idx]
            if ps.lowest_card and ps.highest_card:
                low_rank = ps.lowest_card.rank
                high_rank = ps.highest_card.rank
                # Ranks are 1-13, indices 0-12
                # e.g. 7 is played. low=7, high=7. range(7, 8) -> 7. idx=6.
                for r in range(low_rank, high_rank + 1):
                    state[1, s_idx, r - 1] = 1.0
                    
        # Ch 2: Legal Moves
        # Which card *on the board* accepts a move? No, "Legal Moves Mask" implies
        # which cards *I can play*. 
        # "1=可出, 0=不可"
        # This usually means all VALID moves in the game (not just cards I hold).
        # Actually, usually in AlphaZero for card games:
        # It's a mask of valid actions over the action space.
        # Action space = 52 cards.
        # So if dropping Heart 6 is valid (cuz Heart 7 is on board), then Heart 6 position is 1.
        # Wait, does it mean "Cards I HOLD that are legal" or "Cards that WOULD be legal if I held them"?
        # Usually it's "Legal Actions". My action is "Play Card X".
        # So it must be cards I HOLD and can play.
        # Let's verify with "Legal Moves Mask". 
        # If I don't hold the card, I can't play it.
        # So it's effectively "My Hand" AND "Board Logic valid".
        valid_moves = self.get_all_valid_moves(player_number)
        for c in valid_moves:
            s_idx, r_idx = c.to_tensor_index()
            state[2, s_idx, r_idx] = 1.0
            
        # Ch 3-5: Opponent Covered Card Count (Normalized)
        # Relative positions: 
        # P1 (Next): (player_idx + 1) % 4
        # P2 (Opp): (player_idx + 2) % 4
        # P3 (Prev): (player_idx + 3) % 4
        for offset in range(1, 4):
            opp_idx = (player_idx + offset) % self.num_players
            count = self.hands[opp_idx].get_card_count()
            val = count / 13.0
            # Fill the entire channel with this value? Or just Scalar?
            # CNN input expects spatial. Usually we fill the whole plane.
            state[2 + offset, :, :] = val
            
        # Ch 6-8: Opponent Pass Record (Suit-level missing information)
        for offset in range(1, 4):
            opp_idx = (player_idx + offset) % self.num_players
            opp_passes = self.pass_record[opp_idx]
            for s_idx in range(4):
                if opp_passes[s_idx]:
                    state[5 + offset, s_idx, :] = 1.0
                    
        # Ch 9: Dealer Indicator
        # "全 1=我是莊家"
        if player_number == self.dealer_number:
            state[9, :, :] = 1.0
            
        # Ch 10: Turn Count
        # Normalized t/52
        state[10, :, :] = self.turn_count / 52.0
        
        return state
        
    def get_belief_target(self, observer_player: int) -> list:
        """
        Returns a flat list representing a 3x4x13 tensor of opponent hands and covered cards.
        """
        import numpy as np
        target = np.zeros((3, 4, 13), dtype=np.float32)
        obs_idx = observer_player - 1
        opp_indices = [i for i in range(4) if i != obs_idx]
        
        for opp, p_idx in enumerate(opp_indices):
            # Hand
            for c in self.hands[p_idx].cards:
                s_idx, r_idx = c.to_tensor_index()
                target[opp, s_idx, r_idx] = 1.0
                
            # Covered
            for c in self.covered_cards[p_idx]:
                s_idx, r_idx = c.to_tensor_index()
                target[opp, s_idx, r_idx] = 1.0
                
        return target.flatten().tolist()

