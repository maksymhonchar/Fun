public class DLNode {
	private int element;
	private DLNode next;
	private DLNode prev;

	public DLNode() {
		this(0, null, null);
	}

	public DLNode(int e, DLNode p, DLNode n) {
		this.element = e;
		this.prev = p;
		this.next = n;
	}

	public void setElement(int element) {
		this.element = element;
	}

	public void setNext(DLNode n) {
		this.next = n;
	}

	public void setPrev(DLNode p) {
		this.prev = p;
	}

	public int getElement() {
		return (this.element);
	}

	public DLNode getNext() {
		return (this.next);
	}

	public DLNode getPrev() {
		return (this.prev);
	}

}
