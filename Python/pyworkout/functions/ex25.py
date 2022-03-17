import xml.etree.ElementTree as XMLElementTree


def get_xml_element(
    tag: str,
    text: str = '',
    **attrib
) -> bytes:
    xml_element = XMLElementTree.Element(tag, attrib=attrib)
    xml_element.text = text
    return XMLElementTree.tostring(xml_element)


def get_xml_element_v2(
    tag: str,
    text: str = '',
    **attrib
) -> str:
    attrib_str = ''.join([
        f' {name}="{value}"'
        for name, value in attrib.items()
    ])
    return f'<{tag}{attrib_str}>{text}</{tag}>'


def main():
    print(get_xml_element_v2('foo'))
    print(get_xml_element_v2('foo', 'bar'))
    print(get_xml_element_v2('foo', 'bar', a='1', b='2', c='3'))


if __name__ == '__main__':
    main()
