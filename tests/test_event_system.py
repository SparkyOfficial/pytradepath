"""
Unit tests for the event system of the pytradepath framework.
"""

import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.event import EventType, MarketEvent, SignalEvent, OrderEvent, FillEvent


class TestEventSystem(unittest.TestCase):
    
    def test_market_event_creation(self):
        """Test creation of MarketEvent"""
        symbol = "AAPL"
        data = {"price": 150.0, "volume": 1000}
        event = MarketEvent(symbol, data)
        
        self.assertEqual(event.type, EventType.MARKET)
        self.assertEqual(event.symbol, symbol)
        self.assertEqual(event.data, data)
        self.assertIsNotNone(event.timestamp)
    
    def test_signal_event_creation(self):
        """Test creation of SignalEvent"""
        symbol = "AAPL"
        signal_type = "BUY"
        strength = 0.8
        event = SignalEvent(symbol, signal_type, strength)
        
        self.assertEqual(event.type, EventType.SIGNAL)
        self.assertEqual(event.symbol, symbol)
        self.assertEqual(event.signal_type, signal_type)
        self.assertEqual(event.strength, strength)
        self.assertIsNotNone(event.timestamp)
    
    def test_order_event_creation(self):
        """Test creation of OrderEvent"""
        symbol = "AAPL"
        order_type = "MARKET"
        quantity = 100
        direction = "BUY"
        event = OrderEvent(symbol, order_type, quantity, direction)
        
        self.assertEqual(event.type, EventType.ORDER)
        self.assertEqual(event.symbol, symbol)
        self.assertEqual(event.order_type, order_type)
        self.assertEqual(event.quantity, quantity)
        self.assertEqual(event.direction, direction)
        self.assertIsNotNone(event.timestamp)
    
    def test_fill_event_creation(self):
        """Test creation of FillEvent"""
        symbol = "AAPL"
        quantity = 100
        direction = "BUY"
        fill_price = 150.0
        commission = 1.0
        event = FillEvent(symbol, quantity, direction, fill_price, commission)
        
        self.assertEqual(event.type, EventType.FILL)
        self.assertEqual(event.symbol, symbol)
        self.assertEqual(event.quantity, quantity)
        self.assertEqual(event.direction, direction)
        self.assertEqual(event.fill_price, fill_price)
        self.assertEqual(event.commission, commission)
        self.assertAlmostEqual(event.cost, quantity * fill_price + commission)


if __name__ == '__main__':
    unittest.main()