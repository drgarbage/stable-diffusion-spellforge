#!/usr/bin/env python



def preload(parser):
    parser.add_argument(
        "--spellforge-apikey",
        type=str,
        help="API key to access generation",
        default=None,
    )