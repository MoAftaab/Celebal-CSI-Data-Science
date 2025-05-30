#!/usr/bin/env python3
"""
Singly Linked List Implementation

This module implements a singly linked list data structure using Object-Oriented
Programming principles. It includes classes for nodes and the linked list itself,
with methods for adding nodes, printing the list, and deleting nodes.

Author: Mohd Aftaab
"""


class Node:
    """
    Represents a node in a singly linked list.
    
    Each node contains data and a reference to the next node in the sequence.
    """
    
    def __init__(self, data):
        """
        Initialize a new node with the given data.
        
        Args:
            data: The value to be stored in the node
        """
        self.data = data
        self.next = None


class LinkedList:
    """
    Implements a singly linked list data structure.
    
    The linked list maintains a reference to the head node and provides
    methods to manipulate the list.
    """
    
    def __init__(self):
        """Initialize an empty linked list."""
        self.head = None
    
    def append(self, data):
        """
        Add a new node with the given data to the end of the list.
        
        Args:
            data: The value to be stored in the new node
        """
        new_node = Node(data)
        
        # If the list is empty, set the new node as the head
        if self.head is None:
            self.head = new_node
            return
        
        # Otherwise, traverse to the end of the list
        current = self.head
        while current.next:
            current = current.next
        
        # Set the next reference of the last node to the new node
        current.next = new_node
    
    def print_list(self):
        """
        Print all elements in the linked list.
        
        If the list is empty, prints a message indicating so.
        """
        if self.head is None:
            print("Linked list is empty")
            return
        
        current = self.head
        while current:
            print(current.data, end=" -> " if current.next else "\n")
            current = current.next
    
    def delete_nth(self, n):
        """
        Delete the nth node from the linked list (1-based indexing).
        
        Args:
            n: The position of the node to delete (1-based index)
            
        Raises:
            ValueError: If the list is empty or index is out of range
        """
        if self.head is None:
            raise ValueError("Cannot delete from an empty list")
        
        # Special case: deleting the head
        if n == 1:
            self.head = self.head.next
            return
        
        # Find the node before the one to be deleted
        current = self.head
        count = 1
        
        while current and count < n - 1:
            current = current.next
            count += 1
        
        # Check if we reached the end of the list or if n is out of range
        if current is None or current.next is None:
            raise ValueError(f"Index {n} is out of range")
        
        # Delete the nth node by updating the next reference
        current.next = current.next.next


# Testing the implementation
if __name__ == "__main__":
    # Create a linked list
    linked_list = LinkedList()
    
    # Add elements to the list
    linked_list.append(10)
    linked_list.append(20)
    linked_list.append(30)
    linked_list.append(40)
    linked_list.append(50)
    
    # Print the original list
    print("Original linked list:")
    linked_list.print_list()
    
    # Delete the 3rd node (30)
    try:
        linked_list.delete_nth(3)
        print("\nLinked list after deleting 3rd node:")
        linked_list.print_list()
    except ValueError as e:
        print(f"Error: {e}")
    
    # Try to delete an out-of-range node
    try:
        linked_list.delete_nth(10)
    except ValueError as e:
        print(f"\nError: {e}")
    
    # Delete all remaining nodes
    print("\nDeleting all nodes one by one:")
    try:
        while True:
            linked_list.delete_nth(1)
            linked_list.print_list()
    except ValueError as e:
        print(f"Error: {e}")
    
    # Try to delete from empty list
    try:
        linked_list.delete_nth(1)
    except ValueError as e:
        print(f"\nError: {e}") 