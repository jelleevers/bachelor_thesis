using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace cgn2train
{
    class Program
    {

        static void Main(string[] args)
        {
            if (args.Length < 7)
                return;

            string cgn_path = args[0];
            string out_path = args[1];
            int word_count = 1200000;
            int.TryParse(args[2], NumberStyles.Integer, CultureInfo.InvariantCulture, out word_count);
            int vocab_size = 10000;
            int.TryParse(args[3], NumberStyles.Integer, CultureInfo.InvariantCulture, out vocab_size);
            float valid_fract = 0.1f;
            float.TryParse(args[4], NumberStyles.Float, CultureInfo.InvariantCulture, out valid_fract);
            float test_fract = 0.1f;
            float.TryParse(args[5], NumberStyles.Float, CultureInfo.InvariantCulture, out test_fract);
            int rand_seed = 1;
            int.TryParse(args[6], NumberStyles.Integer, CultureInfo.InvariantCulture, out rand_seed);

            Console.WriteLine("CGN2Train parameters:");
            Console.WriteLine("Word count       = {0}", word_count);
            Console.WriteLine("Vocabulary size  = {0}", vocab_size);
            Console.WriteLine("Valid fraction   = {0}", valid_fract);
            Console.WriteLine("Test fraction    = {0}", test_fract);
            Console.WriteLine("Random seed      = {0}", rand_seed);
            Console.WriteLine();

            string[] cgn_files = Directory.GetFiles(cgn_path, "*.txt");

            Dictionary<string, int> cgn_vocabulary = new Dictionary<string, int>();

            int cgn_word_count = 0;
            int[] cgn_file_word_counts = new int[cgn_files.Length];
            for (int i = 0; i < cgn_files.Length; i++)
            {
                cgn_file_word_counts[i] = _read_word_count(cgn_files[i], cgn_vocabulary);
                cgn_word_count += cgn_file_word_counts[i];
            }

            Console.WriteLine("CGN data info:");
            Console.WriteLine("Word count       = {0}", cgn_word_count);
            Console.WriteLine("Vocabulary size  = {0}", cgn_vocabulary.Count);
            Console.WriteLine();

            word_count = Math.Max(0, Math.Min(word_count, (int)(0.90f * cgn_word_count)));

            float file_word_count_fract = (float)word_count / cgn_word_count;

            Random rnd_obj = new Random(rand_seed);

            Dictionary<string, int> data_vocabulary = new Dictionary<string, int>();
            List<Tuple<int, string>> data_lines = new List<Tuple<int, string>>();
            List<string> train_lines = new List<string>();
            List<string> valid_lines = new List<string>();
            List<string> test_lines = new List<string>();

            word_count = 0;
            for (int i = 0; i < cgn_files.Length; i++)
            {
                int data_word_count = _read_data_lines(cgn_files[i], (int)(cgn_file_word_counts[i] * file_word_count_fract), data_vocabulary, data_lines, rnd_obj);

                _split_train_lines(data_lines, data_word_count, train_lines, valid_lines, test_lines, valid_fract, test_fract);

                word_count += data_word_count;

                data_lines.Clear();
            }

            int original_vocab_size = data_vocabulary.Count;
            float vocab_coverage = _reduce_vocabulary(ref data_vocabulary, train_lines, vocab_size);
            int words_filtered_count = 0;

            words_filtered_count += _filter_train_lines(train_lines, data_vocabulary);
            words_filtered_count += _filter_train_lines(valid_lines, data_vocabulary);
            words_filtered_count += _filter_train_lines(test_lines, data_vocabulary);

            Console.WriteLine("Train data info:");
            Console.WriteLine("Word count       = {0}", word_count);
            Console.WriteLine("Orig vocab size  = {0}", original_vocab_size);
            Console.WriteLine("Vocab coverage   = {0}", vocab_coverage);
            Console.WriteLine("Words filtered   = {0}", words_filtered_count);
            Console.WriteLine();

            string train_file = Path.Combine(out_path, "train.txt");
            string valid_file = Path.Combine(out_path, "valid.txt");
            string test_file = Path.Combine(out_path, "test.txt");

            _write_train_lines(train_file, train_lines);
            _write_train_lines(valid_file, valid_lines);
            _write_train_lines(test_file, test_lines);

            Console.WriteLine("Done!");
            Console.ReadKey();
        }

        static int _read_word_count(string file, Dictionary<string, int> vocab)
        {
            int word_count = 0;

            string[] lines = File.ReadAllLines(file);
            foreach (string line in lines)
            {
                string[] words = line.Replace("LACH", "").ToLowerInvariant().Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                if (words.Length >= 1)
                {
                    foreach (string word in words)
                        vocab[word] = (vocab.ContainsKey(word) ? vocab[word] : 0) + 1;
                    word_count += words.Length + 1;

                    vocab["<eos>"] = (vocab.ContainsKey("<eos>") ? vocab["<eos>"] : 0) + 1;
                }
            }

            return word_count;
        }

        static int _read_data_lines(string file, int min_word_count, Dictionary<string, int> vocab, List<Tuple<int, string>> data_lines, Random rnd_obj)
        {
            string[] lines = File.ReadAllLines(file);
            int line_count = lines.Length;
            int line_start = rnd_obj.Next(line_count);

            int word_count = 0;
            while (word_count < min_word_count)
            {
                string[] words = lines[line_start].Replace("LACH", "").ToLowerInvariant().Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                if (words.Length >= 1)
                {
                    foreach (string word in words)
                        vocab[word] = (vocab.ContainsKey(word) ? vocab[word] : 0) + 1;

                    data_lines.Add(new Tuple<int, string>(words.Length + 1, string.Join(" ", words) + " <eos>"));

                    word_count += words.Length + 1;

                    vocab["<eos>"] = (vocab.ContainsKey("<eos>") ? vocab["<eos>"] : 0) + 1;
                }

                line_start = (line_start + 1) % line_count;
            }

            return word_count;
        }

        static void _split_train_lines(List<Tuple<int, string>> data_lines, int data_words, List<string> train_lines, List<string> valid_lines, List<string> test_lines, float valid_fract, float test_fract)
        {
            int valid_words = (int)Math.Ceiling(data_words * valid_fract);
            int test_words = (int)Math.Ceiling(data_words * test_fract);

            int data_lines_idx = 0;

            while (valid_words > 0 && data_lines_idx < data_lines.Count)
            {
                valid_lines.Add(data_lines[data_lines_idx].Item2);
                valid_words -= data_lines[data_lines_idx].Item1;
                data_lines_idx++;
            }

            while (test_words > 0 && data_lines_idx < data_lines.Count)
            {
                test_lines.Add(data_lines[data_lines_idx].Item2);
                test_words -= data_lines[data_lines_idx].Item1;
                data_lines_idx++;
            }

            while (data_lines_idx < data_lines.Count)
            {
                train_lines.Add(data_lines[data_lines_idx].Item2);
                data_lines_idx++;
            }
        }

        static float _reduce_vocabulary(ref Dictionary<string, int> vocab, List<string> train_lines, int new_vocab_size)
        {
            string train_txt = string.Join("", train_lines);

            int unk_count = vocab.ContainsKey("<unk>") ? vocab["<unk>"] : 0;
            int eos_count = vocab.ContainsKey("<eos>") ? vocab["<eos>"] : 0;

            vocab.Remove("<unk>");
            vocab.Remove("<eos>");

            HashSet<string> train_words = new HashSet<string>();
            
            foreach(string train_line in train_lines)
            {
                string[] words = train_line.Split(' ');
                foreach (string word in words)
                    train_words.Add(word);
            }

            List<Tuple<string, int, int>> vocab_lst_ext = vocab.Select(x => new Tuple<string, int, int>(x.Key, x.Value, train_words.Contains(x.Key) ? 1 : 0)).ToList();

            int ns_word_count = vocab_lst_ext.Sum(x => x.Item2);

            vocab_lst_ext.Sort((a, b) => b.Item2 != a.Item2 ? b.Item2.CompareTo(a.Item2) : b.Item3.CompareTo(a.Item3));

            vocab_lst_ext = vocab_lst_ext.Take(new_vocab_size - 2).ToList();

            int reduced_ns_word_count = vocab_lst_ext.Sum(x => x.Item2);

            unk_count += ns_word_count - reduced_ns_word_count;

            vocab = vocab_lst_ext.ToDictionary(x => x.Item1, x => x.Item2);

            train_lines.Insert(0, string.Join(" ", vocab_lst_ext.FindAll(x => x.Item3 == 0).Select(x => x.Item1).ToArray()) + " <eos>");

            vocab.Add("<unk>", unk_count);
            vocab.Add("<eos>", eos_count);

            return (float)reduced_ns_word_count / ns_word_count;
        }

        static int _filter_train_lines(List<string> train_lines, Dictionary<string, int> vocab)
        {
            int words_removed = 0;

            for (int n = 0; n < train_lines.Count; n++)
            {
                string[] words = train_lines[n].Split(' ');

                for (int i = 0; i < words.Length; i++)
                    if (!vocab.ContainsKey(words[i]))
                    {
                        words[i] = "<unk>";
                        words_removed++;
                    }

                train_lines[n] = string.Join(" ", words);
            }

            return words_removed;
        }

        static void _write_train_lines(string file, List<string> train_lines)
        {
            File.WriteAllText(file, (" " + string.Join(" ", train_lines) + " ").Replace(" <eos> ", " \n "), new UTF8Encoding(false));
        }

    }
}
