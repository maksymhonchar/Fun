package mainpackage;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class ReadCSV {

	public static void main(String[] args) {
		ReadCSV obj = new ReadCSV();
		String data[][] = new String[10][10];
		data = obj.run();
		System.out.println(data[0][0] + data[0][1]);
	}

	String[][] run() {
		String[][] dataToRet = new String[10][10];
		String csvFilePath = "C:/Users/maxgo/Desktop/StartupDirector.csv";
		BufferedReader br = null;
		String line = "";
		String csvSplitBy = ",";
		int index = 0;
		
		try {
			br = new BufferedReader(new FileReader(csvFilePath));
			while ((line = br.readLine()) != null) {
				String[] data = line.split(csvSplitBy);
				dataToRet[index++] = data;
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (br != null) {
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		return (dataToRet);
	}
}
