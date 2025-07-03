
def join(separator, items):
    """
    This function takes an arbitrary object 'separator' and a list 'items',
    then returns a new list where every two consecutive elements are separated
    by the 'separator' object.

    Args:
    separator: Any object that will be used as the separator between elements.
    items: A list of elements where the separator will be inserted between consecutive elements.

    Returns:
    A new list with the elements separated by 'separator'.
    """
    if not items:  # If the list is empty, return an empty list
        return []

    # Start with the first element, then add separator and next element repeatedly
    result = [items[0]]
    for item in items[1:]:
        result.append(separator)
        result.append(item)

    return result

if __name__ == '__main__':
    # Example usage:
    separator = ["X"]
    items = [1, 2, 3, 4]
    joined_list = join(separator, items)
    print(joined_list)  # Output: [1, ['X'], 2, ['X'], 3, ['X'], 4]
    separator.append("Y")
    print(joined_list)
