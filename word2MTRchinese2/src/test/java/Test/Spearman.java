package Test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.hankcs.hanlp.HanLP;
import com.hankcs.hanlp.dictionary.py.Pinyin;

public class Spearman {

	ArrayList<ArrayList<String>> simword = new ArrayList<ArrayList<String>>(); // ��ѵ���ļ��ж�ȡ������
	ArrayList<Double> trainwordsim = new ArrayList<Double>(); // ����ѵ���ļ��еĴʶԣ�������Ĵ����ƶȣ�˳����simwordһ��
	ArrayList<Integer> loss = new ArrayList<Integer>(); // �����ļ��к��ԵĴʶԣ��洢���ǴʶԵ�id

	Map<String, Integer> words = new HashMap<>(); 
	ArrayList<ArrayList<Double>> C = new ArrayList<ArrayList<Double>>(); 
	Map<String, Integer> twords = new HashMap<>(); 
	ArrayList<ArrayList<Double>> tC = new ArrayList<ArrayList<Double>>();
	Map<String, Integer> pinyins = new HashMap<>(); // pinyin map
	ArrayList<ArrayList<Double>> P = new ArrayList<ArrayList<Double>>();

	int dim = 100;

	String trainfile;// �������ļ�
	String evaluatefile;// �����ļ�����240.txt

	String filesplit;
	String yunmu=" ang1、ang2、ang3、ang4、eng1、eng2、eng3、eng4、ing1、ing2、ing3、ing4、ong1、ong2、ong3、ong4"
			+ "b、p、m、f、d、t、n、l、g、k、h、j、q、x、zh、ch、sh、z、c、s、y、w、r"
			+ "、a1、a2、a3、a4、o1、o2、o3、o4、e1、e2、e3、e4、i1、i2、i3、i4、u1、u2、u3、u4、v1、v2、v3、v4、ai1、ai2、ai3、ai4、"
			+ "ei1、ei2、ei3、ei4、ui1、ui2、ui3、ui4、ao1、ao2、ao3、ao4、 ou1、ou2、ou3、ou4、 iu1、iu2、iu3、iu4、"
			+ "ie1、ie2、ie3、ie4、ve1、ve2、ve3、ve4、 er1、er2、er3、er4、 an1、 an2、 an3、 an4、en1、en2、en3、en4、"
			+ "in1、in2、in3、in4、un1、un2、un3、un4、vn1、vn2、vn3、vn4、wu1、wu2、wu3、wu4、ue1、ue2、ue2、ue4";

	public Spearman(String file1, String file2) {
		this.trainfile = file1;
		this.evaluatefile = file2;
	}

	public double computeSpearman() {

		readVec();
		readSimword(evaluatefile);
		trainWordsSimiliar();

		BigDecimal bd;
		ArrayList<WordPair> trainword = new ArrayList<WordPair>(); // �洢����ѵ���õ��ĴʶԵ������Ϣ
		ArrayList<WordPair> fileword = new ArrayList<WordPair>(); // �����ļ��еõ��Ĵʶ���Ϣ

		for (int i = 0; i < trainwordsim.size(); i++) {
			bd = new BigDecimal(trainwordsim.get(i));
			double t = bd.setScale(3, BigDecimal.ROUND_HALF_UP).doubleValue();
			trainword.add(new WordPair(i, t));
		}

		int jk = 0;
		for (int i = 0; i < simword.size(); i++) {
			if (!loss.contains(i)) {
				fileword.add(new WordPair(jk++, Double.valueOf(simword.get(i).get(2))));
			}
		}

		Collections.sort(trainword);
		Collections.sort(fileword);

		calcuRank(trainword);
		calcuRank(fileword);

		/*
		 * for (int i = 0; i < fileword.size(); i++) {
		 * System.out.println(fileword.get(i).id + " " + fileword.get(i).sim + " " +
		 * fileword.get(i).rank); }
		 * 
		 * System.out.println("```````````````````````````````````````````````````");
		 * 
		 * for (int i = 0; i < trainword.size(); i++) {
		 * System.out.println(trainword.get(i).id + " " + trainword.get(i).sim + " " +
		 * trainword.get(i).rank); }
		 */

		double d = 0;
		for (int i = 0; i < fileword.size(); i++) {
			for (int j = 0; j < trainword.size(); j++) {
				if (fileword.get(i).id == trainword.get(j).id) {
					d = d + (fileword.get(i).rank - trainword.get(j).rank)
							* (fileword.get(i).rank - trainword.get(j).rank);
					break;
				}
			}
		}

		double size = fileword.size();
		double spearman = 1 - (6 * d) / (size * (size * size - 1));
		System.out.println("忽略的词" + loss.size());
		System.out.println("参与训练的词" + trainword.size());
		return spearman * 100;
	}

	public void calcuRank(ArrayList<WordPair> wordpairs) {
		// ����ȼ�
		double count = 1.0;
		double sum = 1.0;

		for (int i = 0; i < wordpairs.size(); i++) {
			if (i + 1 < wordpairs.size() && wordpairs.get(i).sim == wordpairs.get(i + 1).sim) {
				count = count + 1;
				sum = sum + i + 2;
			} else {
				double t = sum / count;
				while (count > 0) {
					wordpairs.get((int) (i - count + 1)).rank = t;
					count--;
				}
				count = 1.0;
				sum = i + 2;
			}
		}

	}

	void trainWordsSimiliar() {

		double s1, s2;
		int numa = 0, numb = 0; // �ʶ��ڴʵ��к������
		double theta1a, theta1b, theta2a, theta2b;
		String worda, wordb;
		Double[] X1a = new Double[dim]; // the first word vec
		Double[] X1b = new Double[dim]; // the second word vec
		Double[] X2a = new Double[dim]; // the first word vec
		Double[] X2b = new Double[dim]; // the second word vec

		for (int pairsize = 0; pairsize < simword.size(); pairsize++) {

			s1 = 0.0;
			s2 = 0.0;
			theta1a = 0.0;
			theta1b = 0.0;
			theta2a = 0.0;
			theta2b = 0.0;
			worda = simword.get(pairsize).get(0);
			wordb = simword.get(pairsize).get(1);

			for (int i = 0; i < dim; i++) {
				X1a[i] = 0.0;
				X1b[i] = 0.0;
				X2a[i] = 0.0;
				X2b[i] = 0.0;
			}

			if (twords.get(worda) == null || twords.get(wordb) == null) {
				loss.add(pairsize);
				continue;
			} else {
				numa = twords.get(worda);
				numb = twords.get(wordb);
				for (int i = 0; i < dim; i++) {
					X1a[i] = X1a[i] + tC.get(numa).get(i);
					X1b[i] = X1b[i] + tC.get(numb).get(i);
				}
			}

			double charsizea = 0;
			double charsizeb = 0;
			List<Pinyin> pya = HanLP.convertToPinyinList(worda);
			List<Pinyin> pyb = HanLP.convertToPinyinList(wordb);
			charsizea = worda.length();
			charsizeb = wordb.length();

			
			String s;
			
			for (Pinyin pinyin : pya) {
				if (pinyin == null) {
					System.out.println("worda pinyin not exit");
					continue;
				}
				s= pinyin.toString();
				int indexp = pinyins.get(s);
				for (int k = 0; k < dim; k++) {
					X1a[k] = X1a[k] + P.get(indexp).get(k);
				}
			}
			
			
			
			for (Pinyin pinyin : pyb) {
				if (pinyin == null) {
					System.out.println("wordb pinyin not exit");
					continue;
				}
				s= pinyin.toString();
				int indexp = pinyins.get(s);
				for (int k = 0; k < dim; k++) {
					X1a[k] = X1a[k] + P.get(indexp).get(k);
				}
			}
			

			for (int j = 0; j < dim; j++) {
				theta1a = theta1a + X1a[j] * X1a[j];
				s1 = s1 + X1a[j] * X1b[j];
				theta1b = theta1b + X1b[j] * X1b[j];
				
				theta2a = theta2a + X2a[j] * X2a[j];
				s2 = s2 + X2a[j] * X2b[j];
				theta2b = theta2b + X2b[j] * X2b[j];
			}
			s1 = (double)(s1 / (Math.sqrt(theta1b) * Math.sqrt(theta1a)));
			s2 = s2 / (Math.sqrt(theta2b) * Math.sqrt(theta2a));
			trainwordsim.add(s1);
		}
	}

	public void readVec() {
		// ��ȡ��ѵ���õĴ�����
		try {
			InputStreamReader input = new InputStreamReader(new FileInputStream(trainfile), "utf-8");
			BufferedReader read = new BufferedReader(input);
			String line;
			String[] factors;
			int num = 0, pnum = 0, tnum = 0;
			line = read.readLine();
			factors = line.split(" ");
//			wa = Double.parseDouble(factors[0]);
//			wb = Double.parseDouble(factors[1]);
			while ((line = read.readLine()) != null) {
				if (line.equals("the word vector is ")) {
					continue;
				}
				if (line.equals("the pinyin vector is ") || line.equals("the temp vector is ")) {
					break;
				}
				factors = line.split(" ");
				words.put(factors[0], num);
				ArrayList<Double> vec = new ArrayList<Double>();
				for (int i = 1; i <= dim; i++) {
					vec.add(Double.valueOf(factors[i]));
				}
				C.add(vec);
				num++;
			}
			
			while ((line = read.readLine()) != null) {
				if (line.equals("the pinyin vector is ")) {
					break;
				}
				factors = line.split(" ");
				twords.put(factors[0], tnum);
				ArrayList<Double> vec = new ArrayList<Double>();
				for (int i = 1; i <= dim; i++) {
					vec.add(Double.valueOf(factors[i]));
				}
				tC.add(vec);
				tnum++;
			}

			while ((line = read.readLine()) != null) {
				factors = line.split(" ");
				pinyins.put(factors[0], pnum);
				ArrayList<Double> vec = new ArrayList<Double>();
				for (int i = 1; i <= dim; i++) {
					vec.add(Double.valueOf(factors[i]));
				}
				P.add(vec);
				pnum++;
			}
			System.out.println("C size " + C.size() + " " + num);
			System.out.println("tC size " + tC.size() + " " + tnum);
			System.out.println("P size " + P.size() + " " + pnum);
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public void readSimword(String file) {
		// ��ȡsimword��Ĵʶ�
		try {
			InputStreamReader in = new InputStreamReader(new FileInputStream(new File(file)), "utf-8");
			BufferedReader read = new BufferedReader(in);
			String line;
			line = read.readLine();
			String[] split;
			while ((line = read.readLine()) != null) {

				ArrayList<String> temparray = new ArrayList<String>();
				split = line.split("\t");
				for (int i = 0; i < split.length; i++) {
					temparray.add(split[i]);
				}
				simword.add(temparray);

			}
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
