def delist_the_list(items: list):
    for i in range(len(items)):
        items[i] = items[i][0]
    return items
