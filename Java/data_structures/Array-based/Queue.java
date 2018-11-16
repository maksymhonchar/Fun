public class Queue {
	private static final int CAPACITY = 1000;
	private int front;
	private int rear;
	private int N;
	private double[] queueArr;

	public Queue() {
		this.queueArr = new double[CAPACITY];
		this.front = 0;
		this.rear = 0;
		this.N = 0;
	}

	public Queue(int inputSize) {
		this.queueArr = new double[inputSize];
		this.front = 0;
		this.rear = 0;
		this.N = 0;
	}

	public int size() {
		return (N);
	}

	public boolean empty() {
		return (N == 0);
	}

	public void enqueue(double element) {
		if (size() == queueArr.length) {
			appendElement();
		}
		queueArr[rear] = element;
		rear = (rear + 1) % queueArr.length;
		N++;
	}

	public double dequeue() {
		if (empty()) {
			return (0);
		}
		double temp = queueArr[front];
		front = (front + 1) % queueArr.length;
		N--;
		return (temp);
	}

	public void appendElement() {
		double[] queueArr_temp = new double[queueArr.length + 1];
		System.arraycopy(queueArr, front, queueArr_temp, 0, queueArr.length - front);
		System.arraycopy(queueArr, 0, queueArr_temp, queueArr.length - front, rear);
		front = 0;
		rear = queueArr.length;
		queueArr = queueArr_temp;
	}

	/**
	 * Testing function <br>
	 * Print elements of the queue.
	 */
	public void printQueue() {
		if (empty()) {
			System.out.println("Queue is empty.");
			return;
		}
		for (int i = 0; i < N; i++) {
			System.out.printf("%.2f ", queueArr[(front + i) % queueArr.length]);
		}
		System.out.println("");
	}
}
