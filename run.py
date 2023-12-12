import sys

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        import cora
        import imdb
        import enzymes
        import pascal
    
    if 'cora' in args:
        import cora
    if 'imdb' in args:
        import imdb
    if 'enzymes' in args:
        import enzymes
    if 'pascal' in args:
        import pascal
    