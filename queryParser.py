import xml.etree.ElementTree as ET

class QueryXMLParser:
    def __init__(self, xml_path):
        self.query_concepts = []
        xml = ET.parse(xml_path)
        root = xml.getroot()
        topics = root.findall('topic')
        for topic in topics:
            concepts = topic.find('concepts').text.replace("\n"," ").replace("。"," ")
            concepts = concepts.split("、")
            print(concepts)
            self.query_concepts.append(concepts)
