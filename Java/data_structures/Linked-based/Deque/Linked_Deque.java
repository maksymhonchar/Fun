public class Linked_Deque {
	private DLNode header;
	private DLNode trailer;
	private int size;

	public Linked_Deque() {
		this.header = new DLNode();
		this.trailer = new DLNode();
		this.header.setNext(trailer);
		this.header.setPrev(null);
		this.trailer.setPrev(header);
		this.trailer.setNext(null);
		this.size = 0;
	}

	public int size() {
		return (size);
	}

	public boolean empty() {
		return (size == 0);
	}

	public int first() {
		if (empty()) {
			return (0);
		}
		return (header.getNext().getElement());
	}

	public int last() {
		if (empty()) {
			return (0);
		}
		return (trailer.getPrev().getElement());
	}

	public void insertFirst(int element) {
		DLNode second = header.getNext();
		DLNode first = new DLNode(element, header, second);
		second.setPrev(first);
		header.setNext(first);
		size++;
	}

	public void insertLast(int element) {
		DLNode preLast = trailer.getPrev();
		DLNode last = new DLNode(element, preLast, trailer);
		preLast.setNext(last);
		trailer.setPrev(last);
		size++;
	}

	public int removeLast() {
		if (empty()) {
			return (0);
		}
		DLNode last = trailer.getPrev();
		int temp = last.getElement();
		DLNode preLastToLast = last.getPrev();
		trailer.setPrev(preLastToLast);
		preLastToLast.setNext(trailer);
		size--;
		return (temp);
	}

	public int removeFirst() {
		if (empty()) {
			return (0);
		}
		DLNode first = header.getNext();
		int temp = first.getElement();
		DLNode secondToFirst = first.getNext();
		header.setNext(secondToFirst);
		secondToFirst.setPrev(header);
		size--;
		return (temp);
	}
}
