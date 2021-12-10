def ParseCmdLineArguments():
	parser=argparse.ArgumentParser(description='Occupancy Function Training')
	parser.add_argument('--input', type=str, default='./data/HUMBI_256_occ.pkl')
	parser.add_argument('--output', type=str, default='./results/implicit')
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--learning_rate', type=float, default=1.e-3)
	parser.add_argument('--epochs', type=int, default=10)

	return parser.parse_args()