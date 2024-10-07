'''
Unit tests for hotel_classes.py
'''
import unittest
from hotel_classes import Hotel, Customer, Reservation


class TestHotel(unittest.TestCase):
    '''
    Class to test the Hotel class
    '''
    def test_create_hotel(self):
        '''
        Function to evaluate create_hotel
        '''
        hotel = Hotel("Hotel Magno", [101, 102, 201, 202])
        self.assertEqual(hotel.name, "Hotel Magno")
        self.assertEqual(
            hotel.occupied_rooms,
            {101: False, 102: False, 201: False, 202: False}
        )

    def test_reserve_room(self):
        '''
        Function to evaluate reserve_room
        '''
        hotel = Hotel("Hotel Magno", [101, 102, 201, 202])
        hotel.reserve_room(101)
        self.assertEqual(hotel.occupied_rooms[101], True)


class TestCustomer(unittest.TestCase):
    '''
    Class to test the Customer class
    '''
    def test_create_customer(self):
        '''
        Function to evaluate create_customer
        '''
        customer = Customer("Valentin Elizalde")
        self.assertEqual(customer.name, "Valentin Elizalde")

    def test_modify_customer_info(self):
        '''
        Function to evaluate modify_customer_info
        '''
        customer = Customer("Valentin Elizalde")
        customer.modify_info("Pedro Paramo")
        self.assertEqual(customer.name, "Pedro Paramo")


class TestReservation(unittest.TestCase):
    '''
    Class to test the Reservation class
    '''
    def test_create_reservation(self):
        '''
        Function to evaluate create_reservation
        '''
        hotel = Hotel("Hotel Magno", [101, 102, 201, 202])
        reservation_sheet = Reservation(hotel)
        customer = Customer("Pedro Paramo")

        reservation_sheet.create_reservation(hotel, 101, customer)
        self.assertEqual(hotel.occupied_rooms[101], True)
        self.assertEqual(
            reservation_sheet.reservations[f"{hotel.name}_101"],
            customer.name
        )

    def test_cancel_reservation(self):
        '''
        Function to evaluate cancel_reservation
        '''
        hotel = Hotel("Hotel Magno", [101, 102, 201, 202])
        reservation_sheet = Reservation(hotel)
        customer = Customer("Pedro Paramo")

        reservation_sheet.create_reservation(hotel, 101, customer)
        reservation_sheet.cancel_reservation(hotel, 101, customer)
        self.assertEqual(hotel.occupied_rooms[101], False)
        self.assertEqual(
            reservation_sheet.reservations[f"{hotel.name}_101"],
            None
        )


if __name__ == "__main__":
    unittest.main()
