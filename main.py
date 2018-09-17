API_KEY = '2029A39ECAAB627B5F71CF4C3E30F72D'

import dota2api
api = dota2api.Initialise(API_KEY)

match = api.get_match_details(lobby_type=2)

print('match')
print(match)
