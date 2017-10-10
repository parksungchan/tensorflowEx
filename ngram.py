# import ngram
#
# set = ['joe item 1xd qz0123 protest'
#         ,'joe item  1xd qz0123 protest'
#         ,'joe item 1xd qz0  123 protest'
#         ,'joe it   em 1xd qz0123 protest'
#         ,'joe item 2xd qz0123 protest'
#         ,'joe item 2xd qz0243 protest' ]
#
# dataset = ngram.NGram(set)
#
# result = dataset.search('joe item 1xd qz0123 protest')
#
# print(result)