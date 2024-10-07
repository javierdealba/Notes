'''
Classes for hotel.
'''


class Hotel:
    """
    Class for Hotel object.
    """

    def __init__(self, name, rooms):
        self.name = name
        self.occupied_rooms = {room: False for room in rooms}

    def __del__(self):
        print(f"{self.name} instance has been deleted.")

    def display_info(self):
        """
        Method to show object.
        """
        print(f"Hotel: {self.name}")
        for room, occupied in self.occupied_rooms.items():
            if occupied:
                print(f"Room {room}: Occupied")
            else:
                print(f"Room {room}: Available")

    def modify_info(self, name=None, rooms=None):
        """
        Method to modify object.
        """
        if name:
            self.name = name
        if rooms:
            self.occupied_rooms = {room: False for room in rooms}

    def reserve_room(self, room):
        """
        Method to add a reservation in this hotel.
        """
        try:
            if room not in self.occupied_rooms:
                raise ValueError(f"Room {room} does not exist.")
            if self.occupied_rooms[room]:
                raise ValueError(f"Room {room} has a current reservation.")
            self.occupied_rooms[room] = True
        except ValueError as e:
            print(f"Error: {e}")

    def cancel_reservation(self, room):
        """
        Method to cancel a reservation in this hotel.
        """
        try:
            if room not in self.occupied_rooms:
                raise ValueError(f"Room {room} does not exist.")
            if not self.occupied_rooms[room]:
                raise ValueError(
                    f"Room {room} does not have a current reservation."
                    )
            self.occupied_rooms[room] = False
        except ValueError as e:
            print(f"Error: {e}")


class Customer:
    """
    Class for Customer object.
    """
    def __init__(self, name):
        self.name = name

    def __del__(self):
        print(f"{self.name} instance has been deleted.")

    def display_info(self):
        """
        Method to show object.
        """
        print(self.name)

    def modify_info(self, name):
        """
        Method to modify object.
        """
        self.name = name


class Reservation:
    """
    Class for Reservation object
    """
    def __init__(self, hotel):
        self.reservations = {f"{hotel.name}_{room}":
                             None for room in hotel.occupied_rooms}

    def create_reservation(self, hotel, room, customer):
        """
        Method to add a reservation in given hotel room
        by given customer.
        """
        try:
            if room not in hotel.occupied_rooms:
                raise ValueError(f"Room {room} does not exist.")
            if hotel.occupied_rooms[room]:
                raise ValueError(f"Room {room} has a current reservation.")
            hotel.reserve_room(room)
            self.reservations = {f"{hotel.name}_{room}": customer.name}
        except ValueError as e:
            print(f"Error: {e}")

    def cancel_reservation(self, hotel, room, customer):
        """
        Method to cancel a reservation in given hotel room by given customer.
        """
        try:
            if room not in hotel.occupied_rooms:
                raise ValueError(f"Room {room} does not exist.")
            if not hotel.occupied_rooms[room]:
                raise ValueError(f"Room {room} does not \
                                 have a current reservation.")

            if self.reservations[f"{hotel.name}_{room}"] == customer.name:
                hotel.cancel_reservation(room)
                self.reservations = {f"{hotel.name}_{room}": None}
            else:
                raise ValueError("Invalid reservation details.")
        except ValueError as e:
            print(f"Error: {e}")
