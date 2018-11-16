#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent)
{
    this->counter = 0;
    m_button = new QPushButton("MONSTER", this);
    m_button->setGeometry(QRect(QPoint(20, 20), QSize(100, 30)));
    m_label = new QLabel(QString::number(counter), this);
    m_label->setGeometry(QRect(QPoint(60, 60), QSize(30, 20)));
    connect(m_button, SIGNAL(released()), this, SLOT(handleButton()));
}

void MainWindow::handleButton()
{
    counter++;
    m_label->setText(QString::number(counter));
}
