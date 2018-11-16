#include "mainwindow.h"
#include "ui_mainwindow.h"


MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // Initialize a timer.
    this->dataTimer = new QTimer();

    // Set up a COM port.
    this->serialPort = new QSerialPort();
    this->setUpComPort();

    // Create a plot.
    this->makePlot();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::makePlot()
{
    // Generate data for the graph.
    QVector<double> x(100), y(100);
    for(int i = 0; i < 100; i++)
    {
        x[i] = 100*i;
        y[i] = x[i];
    }

    // Set all settings for the graph.
    // Blue graph.
    ui->customPlot->addGraph();
    ui->customPlot->graph(0)->setPen(QPen(QColor(40, 110, 255)));
    ui->customPlot->addGraph();
    ui->customPlot->graph(1)->setPen(QPen(QColor(255, 110, 40)));

    ui->customPlot->axisRect()->setupFullAxesBox();
    ui->customPlot->yAxis->setRange(-1.2, 1.2);

    ui->customPlot->graph(0)->setData(x, y);
    ui->customPlot->xAxis->setLabel("Time, sec");
    ui->customPlot->yAxis->setLabel("Values");

    // Setup X-Axis
    ui->customPlot->xAxis->setTickLabelType(QCPAxis::ltDateTime);
    ui->customPlot->xAxis->setDateTimeSpec(Qt::UTC);
    ui->customPlot->xAxis->setDateTimeFormat("hh:mm:ss");

    connect(ui->customPlot->xAxis, SIGNAL(rangeChanged(QCPRange)),
            ui->customPlot->xAxis2, SLOT(setRange(QCPRange)) );
    connect(ui->customPlot->yAxis, SIGNAL(rangeChanged(QCPRange)),
            ui->customPlot->yAxis2, SLOT(setRange(QCPRange)) );

    connect(dataTimer, SIGNAL(timeout()),
            this, SLOT(realtimeDataSlot()));
    this->dataTimer->start(0); // Refresh as fast as possible.
}

void MainWindow::realtimeDataSlot()
{
    static QTime time(QTime::currentTime());
    // Calculate two new data points.
    double key = time.elapsed()/1000.0;
    static double lastPointKey = 0;
    if(key - lastPointKey > 0.002) // at most add point every 2 ms.
    {
        // Add data to lines.
        ui->customPlot->graph(0)->addData(key, qSin(key)+qrand()/(double)RAND_MAX*1*qSin(key/0.3843));
        ui->customPlot->graph(1)->addData(key, qCos(key)+qrand()/(double)RAND_MAX*0.5*qSin(key/0.4364));
        // Save the last key.
        lastPointKey = key;
    }

    // make key axis range scroll with the data (at a constant range size of 8).
    ui->customPlot->xAxis->setRange(key, 8, Qt::AlignRight);
    ui->customPlot->replot();
}

void MainWindow::setUpComPort()
{
    // Set COM port settings.
    this->serialPort->setPortName("COM4");
    this->serialPort->setBaudRate(QSerialPort::Baud9600);
    this->serialPort->setDataBits(QSerialPort::Data8);
    this->serialPort->setParity(QSerialPort::NoParity);
    this->serialPort->setStopBits(QSerialPort::OneStop);
    this->serialPort->setFlowControl(QSerialPort::NoFlowControl);

    // Open a connection.
    if(!this->serialPort->open(QIODevice::ReadWrite))
    {
        this->ui->content->setText(QString("Error handled: %1").arg(serialPort->errorString()));
        return;
    }

    // Test
    this->ui->content->setText("ok");
}

