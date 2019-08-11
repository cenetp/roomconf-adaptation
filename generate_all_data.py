import generate_cnn_data as gcd

max_room_count = gcd.max_room_count
cnns = ['classificator', 'discriminator']
data_types = ['train', 'test']

# 15000 test and 3000 train for classificator
gcd.generate_data(cnns[0], data_types[0], num_classes=3, amount=5000, mode='no_default_random')
gcd.generate_data(cnns[0], data_types[1], num_classes=3, amount=1000, mode='no_default_random')

# 10000 test and 2000 train for discriminator
gcd.generate_data(cnns[1], data_types[0], num_classes=2, amount=5000, mode='no_default_random')
gcd.generate_data(cnns[1], data_types[1], num_classes=2, amount=1000, mode='no_default_random')
