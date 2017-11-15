#!/usr/bin/env python3

# Simple command line client for network tables
# Add features as needed

import sys
import logging
from networktables import NetworkTables
import time

def printTable( tbl ):
    for k in tbl.getKeys():
        name = NetworkTables.PATH_SEPARATOR.join( ( tbl.path, k ) )
        v = tbl.getValue( k )
        print( '{}: {}'.format( name, v ) )
    
    for st in tbl.getSubTables():
        name = NetworkTables.PATH_SEPARATOR.join( ( tbl.path, st ) )
        printTable( NetworkTables.getTable( name ) )
    return

if __name__ == '__main__':
    import argparse
    from pprint import pprint
    
    parser = argparse.ArgumentParser( description='NetworkTables utility' )
    parser.add_argument( '-s', '--server', default='localhost', help='NetworkTable server address' )
    parser.add_argument( '-t', '--table', default='CameraPublisher', help='Starting table name (default=CameraPublisher)' )
    parser.add_argument( '-v', '--verbose', action='store_true', help='Verbose output' )

    args = parser.parse_args()

    # To see messages from networktables, you must setup logging
    if args.verbose:
        logging.basicConfig( level=logging.DEBUG )
        NetworkTables.enableVerboseLogging()
    else:
        logging.basicConfig( level=logging.INFO )

    NetworkTables.initialize( server=args.server )

    cnt=0
    while not NetworkTables.isConnected():
        if cnt > 0 and cnt % 5 == 0: print( "Failed to connect after {} seconds".format( cnt ) )
        time.sleep(1)
        cnt += 1

    tbl = NetworkTables.getTable( args.table )
    printTable( tbl )
    
