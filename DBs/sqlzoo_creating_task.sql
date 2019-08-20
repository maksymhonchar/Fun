CREATE TABLE student(
  matric_no CHAR(8) PRIMARY KEY,
  first_name VARCHAR(50) NOT NULL,
  last_name VARCHAR(50) NOT NULL,
  date_of_birth DATE
);

INSERT INTO student VALUES ('40001010','Daniel','Radcliffe','1989-07-23');
INSERT INTO student VALUES ('40001011','Emma','Watson','1990-04-15');
INSERT INTO student VALUES ('40001012','Rupert','Grint','1988-10-24');

CREATE TABLE `module`(
  module_code CHAR(8) PRIMARY KEY,
  module_title VARCHAR(50) NOT NULL,
  level INT NOT NULL,
  credits INT NOT NULL DEFAULT 20
);

INSERT INTO module(module_code, module_title, level) VALUES('HUF07101', 'Herbology', 7);
INSERT INTO module(module_code, module_title, level) VALUES('SLY07102', 'Defence Against the Dark Arts', 7);
INSERT INTO module(module_code, module_title, level) VALUES('HUF08102', 'History of Magic', 8);

CREATE TABLE registration(
  matric_no CHAR(8),
  module_code CHAR(8),
  result DECIMAL(4,1),
  PRIMARY KEY (matric_no,module_code),
  FOREIGN KEY (matric_no) REFERENCES student(matric_no),
  FOREIGN KEY (module_code) REFERENCES `module`(module_code)
);

INSERT INTO registration VALUES ('40001010','SLY07102',90);
INSERT INTO registration VALUES ('40001010','HUF07101',40);
INSERT INTO registration VALUES ('40001010','HUF08102',null);

INSERT INTO registration VALUES ('40001011','SLY07102',99);
INSERT INTO registration VALUES ('40001011','HUF08102',null);

INSERT INTO registration VALUES ('40001012','SLY07102',20);
INSERT INTO registration VALUES ('40001012','HUF07101',20);

